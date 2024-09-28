import os
import requests
import json
import uuid
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse

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
        self.voice_model_id = config["model id"]
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
        for config in data["voice"]:
            if voice_name == config["name"]:
                global app_config
                app_config.clear()
                app_config.update(config)
                global chatbot
                chatbot = chatModel(app_config)
                retVal = True
    return retVal

def getAllVoices() -> list:
    retVal = []
    with open("./config.json", "r") as f:
        data = json.load(f)
        for voice in data["voice"]:
            retVal.append(voice["name"])
    return retVal

def generateRandomFilename() -> str:
    random_bytes = os.urandom(16)
    filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + uuid.uuid4().hex + ".mp3"
    return filename

def isFileInDirectory(file_path, directory_path):
    """Checks if a file exists in a directory.

    Args:
        file_path (str): The path to the file.
        directory_path (str): The path to the directory.

    Returns:
        bool: True if the file exists in the directory, False otherwise.
    """

    if os.path.isfile(file_path) and os.path.commonprefix([file_path, directory_path]) == directory_path:
        return True
    else:
        return False

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
        "reference_id": chatbot.voice_model_id,
        "chunk_length": 200,
        "normalize": True,
        "format": "mp3",
        "mp3_bitrate": 64,
        "opus_bitrate": 64,
        "latency": "normal"
    }
    response = requests.post(tts_url, json=payload, headers=headers)
    if response.status_code == 200:
        filename = generateRandomFilename()
        with open(f"output/{filename}", "wb") as f:
            output_bytes = textToSpeech(str(response))
            f.write(output_bytes)
            return filename
    else:
        return "Error! No file generated"

## Start of fastAPI application ##
app = FastAPI()

@app.post("/api/voice/{voice}")
async def setVoice(voice: str):
    voice_list = getAllVoices()
    if voice in voice_list:
        setup(voice)
        return JSONResponse({"status" :"OK"})
    else:
        return JSONResponse({"error": "Voice not found"}, 404)

@app.post("/api/text")
async def generateResponseFromText(request: Request):
    data = await request.json()
    if data is None:
        return JSONResponse({"error": "Invalid JSON input"}, 400)
    prompt = data.get("text")
    msg = chatbot.chat_with_rag(prompt)
    output_path = textToSpeech(msg)
    return JSONResponse({"reply": msg, "file": output_path})

@app.post("/api/audio")
async def generateResponseFromSpeech(request: Request):
    """Generate reply from input prompt of type WAV or MP3 
    """
    data = await request.json()
    if data is None:
        return JSONResponse({"error": "Invalid JSON input"}, 400)
    filename = data.get("file")
    prompt =  speechToText(filename)
    msg = chatbot.chat_with_rag(prompt)
    output_path = textToSpeech(msg)
    return JSONResponse({"reply": msg, "file": output_path})

@app.get("/api/audio/download")
async def returnAudioFileResponse(request: Request):
    data = await request.json()
    if data is None:
        return JSONResponse({"error": "Invalid JSON input"}, 400)
    filename = data.get("file")

    if isFileInDirectory(filename, "output"):
        print(f"{filename} found.")
        file_path = os.path.join(os.getcwd(), "output", filename)
        return FileResponse(file_path)
    else:
        return JSONResponse({"error": "Resource not found"}, 404)
