import os
import json
import uuid
import httpx
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from contextlib import asynccontextmanager

from groq import Groq
from dotenv import load_dotenv, find_dotenv
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
# https://docs.llamaindex.ai/en/stable/examples/llm/groq/

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """handle any user tasks required at startup and shutdown of fastAPI application"""
    ##############################
    # handle user init tasks below
    ##############################
    setup()
    yield
    ############################
    # handle shutdown tasks below
    ############################
    # if there files in all the directories, remove them
    path = os.path.join(os.getcwd(), "output")
    for file_name in os.listdir(path):
        # construct full file path
        file_path = os.path.join(path, file_name)
        if os.path.isfile(file_path):
            print('Deleting file:', file_path)
            os.remove(file_path)

    path = os.path.join(os.getcwd(), "input")
    for file_name in os.listdir(path):
        # construct full file path
        file_path = os.path.join(path, file_name)
        if os.path.isfile(file_path):
            print('Deleting file:', file_path)
            os.remove(file_path)

def setup(voice_name: str = "SortingHat") -> bool:
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

    # make input and output directories if doesn't exists
    in_dir = os.path.join(os.getcwd(), "input")
    if not os.path.exists(in_dir):
        os.mkdir(in_dir)
    out_dir = os.path.join(os.getcwd(), "output")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    return retVal

def get_all_voices() -> list:
    retVal = []
    with open("./config.json", "r") as f:
        data = json.load(f)
        for voice in data["voice"]:
            retVal.append(voice["name"])
    return retVal

def generate_random_filename(file_extension=".mp3") -> str:
    random_bytes = os.urandom(16)
    filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + uuid.uuid4().hex + file_extension
    return filename

def is_file_in_directory(file_name, directory_path):
    """Checks if a file exists in a directory.

    Args:
        file_path (str): The path to the file.
        directory_path (str): The path to the directory.

    Returns:
        bool: True if the file exists in the directory, False otherwise.
    """
    file_path = os.path.join(os.getcwd(), directory_path, file_name)
    if os.path.isfile(file_path):
        return True
    else:
        return False

async def speech_to_text(file_path: str) -> str:
    # Initialize the Groq client
    client = Groq()
    # Create a transcription of the audio file
    with open(file_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(file_path, file.read()), # Required audio file
            model="distil-whisper-large-v3-en", # Required model to use for transcription
            response_format="json",  # Optional
            language="en",  # Optional
            temperature=0.0  # Optional
        )
        # Print the transcription text
        print(transcription.text)
    return transcription.text

async def text_to_speech(prompt: str) -> str:
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

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(tts_url, json=payload, headers=headers)
    except Exception as e:
        # Handle any other unexpected errors
        return JSONResponse({"error": str(e)}, 500)

    if response.status_code == 200:
        filename = generate_random_filename()
        directory = os.path.join(os.getcwd(), "output")
        file_location = os.path.join(directory, filename)
        with open(file_location, "wb+") as f:
            f.write(response.content)
            return filename
    else:
        raise HTTPException(status_code=response.status_code, detail="Error generating audio file")

## Start of fastAPI application ##
app = FastAPI(lifespan=lifespan)

@app.post("/api/voice/{voice}")
async def set_voice(voice: str):
    voice_list = get_all_voices()
    if voice in voice_list:
        setup(voice)
        return JSONResponse({"status" :"OK"})
    else:
        return JSONResponse({"error": "Voice not found"}, 404)

@app.post("/api/text")
async def generate_response_from_text(request: Request):
    data = await request.json()
    if data is None:
        return JSONResponse({"error": "Invalid JSON input"}, 400)
    prompt = data.get("text")
    msg = chatbot.chat_with_rag(prompt)
    output_path = await text_to_speech(msg)
    return JSONResponse({"reply": msg, "output": output_path})

@app.post("/api/audio/upload")
async def generate_response_from_speech(file: UploadFile = File(...)):
    """Generate reply from input prompt of type WAV
    """
    try:
        # Ensure the file format is WAV
        if not file.filename.lower().endswith('.wav'):
            return JSONResponse({"error": "File must be a WAV"}, 400)

        # Generate a unique filename
        filename = generate_random_filename(".wav")
        directory = os.path.join(os.getcwd(), "input")
        file_location = os.path.join(directory, filename)
        with open(file_location, "wb+") as file_object:
            file_object.write(await file.read())

        # Process the audio file
        prompt = await speech_to_text(file_location)
        msg = chatbot.chat_with_rag(prompt)
        output_path = await text_to_speech(msg)
        return JSONResponse({"reply": msg, "output": output_path})
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)

@app.get("/api/audio/download/{file_name}")
async def return_audio_file_response(file_name: str):
    """"
    example:
    GET /api/audio/download/response.mp3
    """
    if is_file_in_directory(file_name, "output"):
        print(f"{file_name} found.")
        file_path = os.path.join(os.getcwd(), "output", file_name)
        return FileResponse(file_path)
    else:
        return JSONResponse({"error": "Resource not found"}, 404)

@app.get("/api/heartbeat")
async def return_heartbeat():
    return JSONResponse({"status": "running"})
