import pickle

def create_pickle(bucket, blob_state):
        # 1. Create a sample dictionary
    sample_data = {
        "blob1.txt": "2023-10-23 15:23:01",
    }
    serialized_data = pickle.dumps(sample_data)

    # Create a new blob or get the existing blob
    blob = bucket.blob(blob_state)
    blob.upload_from_string(serialized_data)
    print(f"Sample data uploaded to '{blob_state}'.")
    return blob