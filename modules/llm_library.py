from modules.vector_search import VectorSearch
from modules.vectorstore import VectorStore
from langchain.chains import ConversationalRetrievalChain
import os
from langchain.embeddings.cohere import CohereEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv



class LLMLibrary:
        
    def __init__(self):
        BASEDIR = os.path.abspath(os.path.dirname("main.py"))
        load_dotenv(os.path.join(BASEDIR, '.env'))
        choere_api_key = os.getenv("COHERE_API_KEY")
        self.embedding_function = CohereEmbeddings(model="embed-multilingual-v2.0", cohere_api_key=choere_api_key)
        self.vectorsearch = VectorSearch()
        self.vectorstore = VectorStore()

    
    def doc_question(self, user_memory, prompt):

        llm = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.0)
        print(f"LLM: {llm}\n")  
        vectordb = self.vectorstore.get_vectordb()
        print(f"Vector DB: {vectordb}\n")
        retriever = vectordb.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "k": 5,
                        "search_distance": 0.6,
                    },
            )

        print(f"Initiating chat conversation memory\n")
        #print(f"Conversation Memory: {memory}\n")
        conversation_chain= ConversationalRetrievalChain.from_llm(
              llm,
              retriever=retriever,
              memory=user_memory,
              rephrase_question=True,
              combine_docs_chain_kwargs={'prompt': prompt},
              return_source_documents=True,
              verbose=True,
        )
        #print(f"Conversation chain: {conversation_chain}\n")
        return conversation_chain