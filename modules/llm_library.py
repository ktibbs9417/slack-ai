from modules.vector_search import VectorSearch
from modules.vectorstore import VectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.memory import VectorStoreRetrieverMemory
from langchain.memory import ConversationBufferMemory
from lunary import LunaryCallbackHandler
import lunary
from langchain.embeddings.cohere import CohereEmbeddings

from dotenv import load_dotenv
import os



from dotenv import load_dotenv



class LLMLibrary:
        
    def __init__(self):
        load_dotenv()
        choere_api_key = os.getenv("COHERE_API_KEY")
        self.embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.cohere_embedding = CohereEmbeddings(model="embed-english-light-v3.0", cohere_api_key=choere_api_key)

        
        self.handler = LunaryCallbackHandler(app_id=os.getenv("LUNARY_APP_ID"))
        self.vectorsearch = VectorSearch()
        self.vectorstore = VectorStore()

    
    def doc_question(self, prompt, user_name):

        llm = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.5, convert_system_message_to_human=True, callbacks=[self.handler], metadata={"user_id": user_name})
        lunary.monitor(llm)
        vectordb = self.vectorstore.get_vectordb()
        print(f"Vector DB: {vectordb}\n")
        retriever = vectordb.as_retriever(
                    search_type="mmr"
            )
        #print(f"Docs: {docs}\n")
        #print(f"Initiating chat conversation memory\n")
        memory =  ConversationBufferMemory(memory_key="chat_history", output_key="answer", return_messages=True, max_token_limit=1024)
        print(f"Conversation Memory: {memory}\n")
        conversation_chain= ConversationalRetrievalChain.from_llm(
              llm,
              retriever=retriever,
              memory=memory,
              combine_docs_chain_kwargs={'prompt': prompt},
              return_source_documents=True,
              verbose=True,
              rephrase_question=True,
        )
        #print(f"Conversation chain: {conversation_chain}\n")
        return conversation_chain
    
    def in_memory_index(self, doc_splits):
        print("Creating Index")
        global vectorstore
        vectorstore = DocArrayInMemorySearch.from_documents(
            doc_splits, 
            embedding=self.cohere_embedding
        )
        print(f"Vector Store: {vectorstore}\n")

    def terraform_question(user_memory, prompt):
        llm = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.0)
        vectordb = vectorstore

        retriever = vectordb.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "k": 5,
                        "search_distance": 0.8,
                    },
            )
        memory = VectorStoreRetrieverMemory(retriever=retriever)
        conversation_chain= LLMChain(
            llm=llm,
            memory=memory,
            prompt=prompt,
            verbose=True,
        )
        #print(f"Conversation chain: {conversation_chain}\n")
        return conversation_chain