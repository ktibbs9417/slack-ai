from utils.matching_engine_utils import MatchingEngineUtils
from google.cloud import storage
import numpy as np
import uuid
import json
import google.auth
from dotenv import load_dotenv
import os

class VectorSearch():
    def __init__(self):
        google.auth.default()
        BASEDIR = os.path.abspath(os.path.dirname("main.py"))
        load_dotenv(os.path.join(BASEDIR, '.env'))
        self.project_id = os.getenv("PROJECT_ID")
        self.region = os.getenv("REGION")
        self.vs_index_name = f"{self.project_id}-{self.region}-vs-index"
        self.vs_embedding_bucket = f"{self.project_id}-{self.region}-vs-bucket"
        self.vs_dimensions = 768

    def create_gcs_bucket(self):
        try:
            self.bucket = storage.Client().get_bucket(self.vs_embedding_bucket)
            print(f"Found existing GCS Bucket: {self.vs_embedding_bucket}")
        except:
            print(f"{self.vs_embedding_bucket} does not exist. Creating bucket.")
            print(f"Creating GCS Bucket: {self.vs_embedding_bucket}")
            self.bucket = storage.Client().create_bucket(self.vs_embedding_bucket, location=self.region)
            print(f"Successfully created GCS Bucket: {self.vs_embedding_bucket}")
            # dummy embedding
            init_embedding = {"id": str(uuid.uuid4()), "embedding": list(np.zeros(self.vs_dimensions))}

            if os.path.exists("init_index"):
                print("utils directory already exists.")
            elif not os.path.exists("init_index"):
                print("init_index directory does not exist. Creating init_index directory.")
                os.makedirs("init_index")
            # dump embedding to a local file
                with open("init_index/embeddings_0.json", "w") as f:
                    json.dump(init_embedding, f)
                blob = self.bucket.blob("init_index/embeddings_0.json")
                blob.upload_from_filename("init_index/embeddings_0.json")
        return

    def create_index(self):

        self.create_gcs_bucket()

        print(f"Creating Vertex Search Index: {self.vs_index_name}")
        vsearch = MatchingEngineUtils(
            self.project_id,
            self.region,
            self.vs_index_name,
        )
        index = vsearch.create_index(
            embedding_gcs_uri = f"gs://{self.vs_embedding_bucket}/init_index",
            dimensions = self.vs_dimensions,
            index_update_method="streaming",
            index_algorithm="tree-ah",
        )
        if index:
            print(f"Successfully created Vertex AI Index: {index.name}")

        index_endpoint = vsearch.deploy_index()
        if index_endpoint:
            print(f"Successfully deployed Vertex AI Index Endpoint: {index_endpoint.name}")
            print(
                f"Index endpoint public domain name: {index_endpoint.public_endpoint_domain_name}"
            )
        for d in index_endpoint.deployed_indexes:
            print(f"{d.id}")
        
        index_id, index_endpoint_id = vsearch.get_index_and_endpoint()

        return index_id, index_endpoint_id, self.project_id, self.region, self.vs_embedding_bucket
    