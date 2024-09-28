import os
import requests
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from groq import Groq
from dotenv import load_dotenv, find_dotenv
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
# https://docs.llamaindex.ai/en/stable/examples/llm/groq/

# !pip install --quiet llama-index-llms-groq
_ = load_dotenv(find_dotenv())  # read local .env file

app_config = dict() # initialize empty dict

class chatModel:
    """
    LLM model with RAG (Retrieval Augmented Generation).

    This class combines a large language model (LLM) with a vector store index to enable more informative and context-aware responses.
    It leverages a retrieval mechanism to access relevant information from a provided data source before generating a response.
    """
    def __init__(self, config: dict):
        self._model = OpenAI("gpt-4o-mini")
        self._system_prompt = config["system prompt"]
        self._init_message = [
            ChatMessage(role="system", content=self._system_prompt),
        ]
        if len(config["folder path"]):
            self._data = SimpleDirectoryReader(input_dir=os.path.join(os.getcwd(), config["folder path"])).load_data()
            self._index = VectorStoreIndex.from_documents(self._data)
            self._chat_engine = self._index.as_chat_engine(
                chat_mode="react",
                llm=self._model,
                chat_history=self._init_message,
                verbose=True
            )

    def chat_with_rag(self, prompt:str) -> str:
        """
        Generates a response using the LLM and RAG.

        Args:
            prompt: The user's input prompt.

        Returns:
            The LLM's response as a string.
        """
        response = self._chat_engine.chat(prompt)
        return str(response)

    def chat(self) -> str:
        """
        Placeholder method (currently not implemented).
        """
        pass

    def clear_message_history(self):
        """
        Resets the chat history of the chat engine.
        """
        self._chat_engine.reset()

chatbot: chatModel

def setup(voice_name: str) -> bool:
    # get app config from config file
    retVal = False
    with open("./config.json", "r") as f:
        data = json.load(f)
        for voice in data["voice"]:
            if voice["name"] == voice_name:
                app_config.update(voice)
                global chatbot
                chatbot = chatModel(app_config)
                retVal = True
    return retVal

def speechToText(input: str) -> str:
    # Initialize the Groq client
    client = Groq()
    # Specify the path to the audio file
    filename = os.path.dirname(__file__) + input # Replace with your audio file!
    # Create a transcription of the audio file
    with open(filename, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(filename, file.read()), # Required audio file
            model="distil-whisper-large-v3-en", # Required model to use for transcription
            response_format="json",  # Optional
            language="en",  # Optional
            temperature=0.0  # Optional
        )
        # Print the transcription text
        print(transcription.text)
    return transcription.text

def textToSpeech(prompt: str) -> str:
    tts_url = "https://api.fish.audio/v1/tts"
    api_key = os.getenv("FISH_API_KEY")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "text": prompt,
        "reference_id": "c944589a55ad450e8109d39cd3ecc488", # model ID
        "chunk_length": 200,
        "normalize": True,
        "format": "mp3",
        "mp3_bitrate": 64,
        "opus_bitrate": 64,
        "latency": "normal"
    }
    resp = requests.post(tts_url, json=payload, headers=headers)
    return resp.content
    with open("response.mp3", "wb") as f:
        output_bytes = textToSpeech(str(response))
        f.write(output_bytes)

    return(os.path.abspath("response.mp3"))

app = FastAPI()

@app.post("/text")
async def getResponseFromText(request: Request):
    data = await request.json()
    if data is None:
        return JSONResponse({"error": "Invalid JSON input"}), 400
    prompt = data.get("text")
    msg = chatbot.chat_with_rag(prompt)
    output_path = textToSpeech(msg)
    return {"text": prompt, "file": output_path}


@app.post("/audio")
async def getResponseFromSpeech(request: Request):
    data = await request.json()
    if data is None:
        return JSONResponse({"error": "Invalid JSON input"}), 400
    filename = data.get("file")
    prompt =  speechToText(filename)
    msg = chatbot.chat_with_rag(prompt)
    output_path = textToSpeech(msg)
    return {"text": prompt, "file": output_path}

try:
    if setup("Sorting Hat") == True:
        # app = Flask(__name__)
        resp = chatbot.chat_with_rag("Summarize the content of the article in the 'data' folder")
        print(resp)
    else:
        raise Exception
except Exception as e:
    # Catch any exception and print its content
    print(f"An exception occurred: {e}")
