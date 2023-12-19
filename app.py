import os
import requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langchain.prompts import PromptTemplate
from langchain.embeddings.cohere import CohereEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
import google.auth
from google.cloud import storage, aiplatform
from dotenv import load_dotenv
from modules.handlers import ChatHandler
from modules.templates import ChatPromptTemplate
from modules.pkl import Pkl

# Load environment variables from .env file
load_dotenv('.env')

# Initializes your app with your bot token and socket mode handler
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))
choere_api_key = os.getenv("COHERE_API_KEY")
embedding_function = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=choere_api_key)
credentials, project = google.auth.default()
aiplatform.init(project=project, location="us-central1")
pkl = Pkl()

#Langchain implementation

conversation_chain= LLMChain(
            llm=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.0),
            memory=ConversationBufferWindowMemory(k=2),
            prompt=ChatPromptTemplate.generic_prompt_template(),
            verbose=True,
        )
conversation_contexts = {}

#Message handler for Slack
@app.message(".*")
def message_handler(message, say, logger):
    print(message)

    user_chain, user_input, history, user_memory, conversation, context_key = ChatHandler.generic_user_message(message, conversation_contexts)   
    output = user_chain.predict(user_input=user_input, history=history, memory=user_memory)
    updated_history = history + f"\nHuman: {user_input}\nAssistant: {output}"
    conversation[context_key]["history"] = updated_history
    conversation[context_key]["memory"] = user_memory
    say(output)

@app.event("app_mention")
def handle_app_mention_events(body, message, say, logger):
    print(body)
    blocks = body['event']['blocks']
    question = blocks[0]['elements'][0]['elements'][1]['text']
    output = conversation_chain.predict(human_input = question)   
    say(output)

@app.command("/doc_question")
def handle_doc_question_command(ack, body, say):
    # Acknowledge the command request
    ack()
    print(body)
    say(f"🤨 {body['text']}")
    question, conversation = ChatHandler.doc_question_command(body, conversation_contexts)
    response = conversation({'question': question})
    print(f"Response: {response['answer']}")

    say(f"🤖 {response['answer']}")
    #conversation = llmlibrary.ask()
    #response = conversation({'question': question})


@app.command("/clear_chat")
def handle_clear_chat_command(ack, body, say):
    # Acknowledge the command request
    ack()

    channel_id = body['channel_id']
    user_id = body['user_id']
    context_key = f"{channel_id}-{user_id}"  # Unique key for each user-channel combination

    # Reset the conversation context for the user-channel combination
    if context_key in conversation_contexts:
        conversation_contexts[context_key] = {
            "memory": ConversationBufferMemory(memory_key="chat_history", output_key="answer", return_messages=True, max_token_limit=1024),
            "history": ""
        }
    conversation = None
    # Send a confirmation message
    say("The chat history has been cleared.")

def upload_file_to_gcs(file_url, file_name):
    storage_client = storage.Client()
    bucket_name = os.getenv("GCS_STORAGE_BUCKET") 
    bucket = storage_client.bucket(os.getenv("GCS_STORAGE_BUCKET") )
    response = requests.get(file_url, headers={'Authorization': f'Bearer {os.environ.get("SLACK_BOT_TOKEN")}'})
    blob = bucket.blob(file_name)
    blob.upload_from_string(response.content)
    pkl.get_blobs()

def file_shared(message, next):
    subtype = message.get("subtype")
    if subtype == "file_share":
        return next()

@app.event("message", middleware=[file_shared])
def handle_message_events(body, say, logger):
    # Check if the message contains a file
    if 'files' in body['event']:
        file_info = body['event']['files'][0]  # Assuming one file per message for simplicity
        file_url = file_info['url_private']
        file_name = file_info['name']

        # Call the function to upload the file to Google Cloud Storage
        try:
            upload_file_to_gcs(file_url, file_name)
            say(f"File {file_name} uploaded to Google Cloud Storage.")
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            say("Failed to upload the file.")

@app.event("message")
def handle_message_events(body, say, logger):
    print("This event does not have a function yet.")
    print(body)
    say("This event does not have a function yet. Please contact the developer with detailed instructions on how to reproduce this error.")

# Start your app
if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()