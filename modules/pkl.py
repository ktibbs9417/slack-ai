import google.auth
import pickle
from google.cloud import storage
from modules.empty_pickle import create_pickle
import os

class Pkl():

    def __init__(self):
        self.credentials, self.project = google.auth.default()
        # Create a list of all the blobs in the Storage Bucket
        self.GCS_STORAGE_BUCKET = os.getenv("GCS_STORAGE_BUCKET")
        self.bucket = storage.Client().get_bucket(self.GCS_STORAGE_BUCKET)
        self.blob_state_name = "blob_state.pkl"
        self.blob_state = self.bucket.blob(self.blob_state_name)

    def get_blobs(self):
         # Get blob_state_data from GCS and if it doesn't exist or is empty, create it with sample data
        from modules.doc_reader_utility import DocumentReader
        self.documentreader = DocumentReader()
        try:
            # Get the Pickle State file from GCS Bucket
            blob_state_data = self.blob_state.download_as_bytes()
        except:
            print("Could not download blob data, possibly it doesn't exist.")
        if not blob_state_data:
            create_pickle(self.bucket, self.blob_state_name)

        # Load the blob_state_data file into a dictionary to process existing blobs
        try:
            prev_blobs = pickle.loads(blob_state_data)
            print(f"Successfully loaded prev_blobs: {prev_blobs.keys()}")
        except Exception as e:
            print(f"Warning: An error occurred while trying to load 'blob_state.pkl': {e}. Initializing with an empty dictionary.")

        # Find new, updated, and deleted blobs
        current_blobs = {blob.name: {'updated': blob.updated, 'uuid': None}for blob in self.bucket.list_blobs() if not blob.name.endswith('.pkl')}
        print(f"Current blobs: {current_blobs}")
        new_blobs = set(current_blobs.keys()) - set(prev_blobs.keys())
        print(f"New blobs: {new_blobs}")
        updated_blobs = {blob for blob in current_blobs if blob in prev_blobs and current_blobs[blob] != prev_blobs[blob]}
        deleted_blobs = set(prev_blobs.keys()) - set(current_blobs.keys())
        print(f"Updated blobs: {updated_blobs}")

        self.documentreader.split_pdf(new_blobs, updated_blobs, deleted_blobs)

        # Update the blob_state_data file with the current_blobs
        #print(f"After processing blobs: {current_blobs}")

        serialized_data = pickle.dumps(current_blobs)
        self.blob_state.upload_from_string(serialized_data)
        print(f"Blob state uploaded to '{self.blob_state_name}'")

        return new_blobs, updated_blobs, deleted_blobs