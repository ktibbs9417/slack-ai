from modules.vector_search import VectorSearch
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv


class VectorStore:
    
    def __init__(self):
        BASEDIR = os.path.abspath(os.path.dirname("main.py"))
        load_dotenv(os.path.join(BASEDIR, '.env'))
        self.embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
            environment=os.getenv("PINECONE_ENV"),  # next to api key in console
            )
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.vectorsearch = VectorSearch()

    def pinecone_vector(self, doc_splits, blob_name):
        # add chunked data to the collection
        try:
            print (f"Adding {blob_name} to Pinecone")

            Pinecone.from_documents(
                documents=doc_splits,
                embedding=self.embedding_function,
                index_name=self.index_name,
            )
            print(f"Successfully added {blob_name} to Pinecone\n")
        except Exception as e:
            print(f"Error: {e}")

    # Load the persistent vectordb
    def get_vectordb(self):        
        return Pinecone.from_existing_index(self.index_name, self.embedding_function)