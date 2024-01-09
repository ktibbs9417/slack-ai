import os
import requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
import google.auth
from google.cloud import storage, aiplatform
from dotenv import load_dotenv
from modules.handlers import ChatHandler
from modules.templates import SlackPromptTemplate
from modules.pkl import Pkl
from modules.tf_reader_utility import TerraformReader
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

# Initializes your app with your bot token and socket mode handler
print(f"Initializing Slack App")
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))
print(f"Initializing Google Auth and Vertex AI")
credentials = google.auth.default()
project = os.getenv("PROJECT_ID")
aiplatform.init(project=project, location="us-central1")
pkl = Pkl()
tf_reader = TerraformReader()
doc_chain = None

#Langchain implementation

conversation_chain= LLMChain(
            llm=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.0),
            memory=ConversationBufferWindowMemory(k=2),
            prompt=SlackPromptTemplate.generic_prompt_template(),
            verbose=True,
        )
conversation_contexts = {}
#Message handler for Slack
@app.message(".*")
def message_handler(message, say, logger):
    print(message)

    user_chain, user_input, history, user_memory, conversation, context_key = ChatHandler.generic_user_message(message, conversation_contexts)   
    response = user_chain.predict(user_input=user_input, history=history, memory=user_memory)
    updated_history = history + f"\nHuman: {user_input}\nAssistant: {response}"
    print(f"(INFO) Generic output: {response} {time.time()}")
    conversation[context_key]["history"] = updated_history
    conversation[context_key]["memory"] = user_memory
    say(f"🤖 {response}")

@app.event("app_mention")
def handle_app_mention_events(body, say, logger):
    print(f"(INFO) App mention: {body} {time.time()}")
    blocks = body['event']['blocks']
    question = blocks[0]['elements'][0]['elements'][1]['text']
    output = conversation_chain.predict(human_input = question)   
    say(f"🤖 {output}")

@app.command("/doc_question")
def handle_doc_question_command(ack, body, say):
    # Acknowledge the command request
    global doc_chain
    ack()
    say(f"🤨 {body['text']}")

    try:
    # Check if conversation_chain is not already set
        if doc_chain is None:
            print("Initializing conversation_chain.")
            doc_chain = ChatHandler.doc_question_command(body)
        else:
            print("doc_chain is already set.")
    except Exception as e:
        # Log the specific exception for better debugging
        print(f"An error occurred: {e}")
        # Optionally, you can reattempt to set conversation_chain if it's critical
        doc_chain = ChatHandler.doc_question_command(body)

    response = doc_chain({'question': body['text']})
    print(f"(DEBUG) Doc Question Response: {response['chat_history']} {time.time()}\n")
    print(f"(DEBUG) Doc Question Full Response: {response} {time.time()}\n")
    print(f"(INFO) User question: {body['text']} {time.time()}\n")
    print(f"(INFO) Doc Question Response answer: {response['answer']} {time.time()}\n")

    say(f"🤖 {response['answer']}")

@app.command("/terraform")
def handle_terraform_command(ack, body, say):
    # Acknowledge the command request
    ack()
    say(f"🤨 {body['text']}")
    text_parts = body['text'].split(' ', 1)
    question = text_parts[1]
    print(f"GCS Path: {text_parts[0]}\n")
    tf_reader.get_tf_state(text_parts[0])

    user_chain, history, user_memory, conversation, context_key = ChatHandler.terraform_question_command(body, conversation_contexts)   
    response = user_chain.predict(question=question, history=history)
    updated_history = history + f"\nHuman: {question}\nAssistant: {response}"
    print(f"(INFO) Terraform question output: {response} {time.time()}")
    conversation[context_key]["history"] = updated_history
    conversation[context_key]["memory"] = user_memory
    say(f"🤖 {response}")


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
    global doc_chain
    doc_chain = None
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