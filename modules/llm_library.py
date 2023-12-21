from modules.vector_search import VectorSearch
from modules.vectorstore import VectorStore
from langchain.chains import ConversationalRetrievalChain
import os
from langchain.chains import LLMChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.memory import VectorStoreRetrieverMemory
from langchain.memory import ConversationBufferMemory


from dotenv import load_dotenv



class LLMLibrary:
        
    def __init__(self):
        self.embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vectorsearch = VectorSearch()
        self.vectorstore = VectorStore()

    
    def doc_question(self, prompt):

        llm = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.0, convert_system_message_to_human=True)
        vectordb = self.vectorstore.get_vectordb()
        print(f"Vector DB: {vectordb}\n")
        retriever = vectordb.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={'score_threshold': 0.8}
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
            embedding=self.embedding_function
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