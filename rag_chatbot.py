import os
import json
import uuid
import httpx
import gradio as gr
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
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
        """
        Initializes the chat model with the provided configuration.

        Args:
            config: A dictionary containing the configuration for the chat model.
        """
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

# Globals
chatbot: chatModel
latest_mp3_response = "" # filename of the latest response from chatbot

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles any user tasks required at startup and shutdown of the FastAPI application.
    """
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
    """
    Sets up the chatbot with the specified voice.

    Args:
        voice_name: The name of the voice to use.

    Returns:
        True if the setup was successful, False otherwise.
    """
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
    """
    Returns a list of all available voices.

    Returns:
        A list of voice names.
    """
    retVal = []
    with open("./config.json", "r") as f:
        data = json.load(f)
        for voice in data["voice"]:
            retVal.append(voice["name"])
    return retVal

def generate_random_filename(file_extension=".mp3") -> str:
    """
    Generates a random filename with the specified extension.

    Args:
        file_extension: The file extension to use.

    Returns:
        A random filename with the specified extension.
    """
    random_bytes = os.urandom(16)
    filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + uuid.uuid4().hex + file_extension
    return filename

def is_file_in_directory(file_name, directory_path):
    """
    Checks if a file exists in a directory.

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

def iterfile(file_path: str):
    """
    Stream an mp3 file in chunks. Python reads file in 8192-byte chunk
    """
    with open(file_path, mode="rb") as file_chunk:
        yield from file_chunk

async def speech_to_text(file_path: str) -> str:
    """
    Transcribes an audio file to text.

    Args:
        file_path: The path to the audio file.

    Returns:
        The transcription of the audio file as a string.
    """
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
    """
    Generates an audio file from text.

    Args:
        prompt: The text to generate audio from.

    Returns:
        The filename of the generated audio file.
    """
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

async def response_from_text(prompt: str) -> tuple:
    """
    Generates a response from text input.

    Args:
        prompt: The text input.

    Returns:
        A tuple containing the generated response and the output audio file path.
    """
    msg = chatbot.chat_with_rag(prompt)
    output_path = await text_to_speech(msg)
    global latest_mp3_response
    latest_mp3_response = output_path
    return (msg, output_path)

async def response_from_speech(file_path) -> tuple:
    """
    Generates a response from speech input.

    Args:
        file_path: The path to the audio file.

    Returns:
        A tuple containing the generated response and the output audio file path.
    """
    prompt = await speech_to_text(file_path)
    msg = chatbot.chat_with_rag(prompt)
    output_path = await text_to_speech(msg)
    global latest_mp3_response
    latest_mp3_response = output_path
    return (msg, output_path)

## Start of fastAPI application ##
app = FastAPI(lifespan=lifespan)

# Add CORS middleware to allow cross-origin requests.
# This is often necessary when your frontend is served from a different domain or port than your API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/api/voice/{voice}")
async def set_voice(voice: str):
    """
    Sets the voice of the chatbot.

    Args:
        voice: The name of the voice to use.

    Returns:
        A JSON response indicating the status of the request.
    """
    voice_list = get_all_voices()
    if voice in voice_list:
        setup(voice)
        return JSONResponse({"status" :"OK"})
    else:
        return JSONResponse({"error": "Voice not found"}, 404)

@app.post("/api/text")
async def generate_response_from_text(request: Request):
    """
    Generates a response from text input.

    Args:
        request: The request object containing the text input.

    Returns:
        A JSON response containing the generated response and the output audio file path.
    """
    data = await request.json()
    if data is None:
        return JSONResponse({"error": "Invalid JSON input"}, 400)
    prompt = data.get("text")
    msg, output_path = await response_from_text(prompt)
    return JSONResponse({"reply": msg, "output": output_path})

@app.post("/api/audio/upload")
async def generate_response_from_speech(file: UploadFile = File(...)):
    """
    Generates a response from speech input.

    Args:
        file: The uploaded audio file.

    Returns:
        A JSON response containing the generated response and the output audio file path.
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
        msg, output_path = await response_from_speech(file_location)
        return JSONResponse({"reply": msg, "output": output_path})
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)

@app.get("/api/audio/download/{file_name}")
async def return_audio_file_response(file_name: str):
    """
    Returns an audio file response.

    Args:
        file_name: The name of the audio file to return.

    Returns:
        A FileResponse containing the audio file.
    """
    if is_file_in_directory(file_name, "output"):
        print(f"{file_name} found.")
        file_path = os.path.join(os.getcwd(), "output", file_name)
        return FileResponse(file_path)
    else:
        return JSONResponse({"error": "Resource not found"}, 404)

@app.get("/api/heartbeat")
async def return_heartbeat():
    """
    Returns a heartbeat response.

    Returns:
        A JSON response indicating the status of the application.
    """
    return JSONResponse({"status": "running"})

@app.get("/api/stream/{file_name}")
async def stream(file_name: str):
    """
    Streams an audio file.

    Args:
        file_name: The name of the audio file to stream.

    Returns:
        A StreamingResponse containing the audio file.
    """
    if is_file_in_directory(file_name, "output"):
        print(f"{file_name} found.")
        file_path = os.path.join(os.getcwd(), "output", file_name)
        # Get the file size
        file_size = os.path.getsize(file_path)
        
        headers = {
            'Accept-Ranges': 'bytes',
            'Content-Type': 'audio/mpeg',
            'Content-Length': str(file_size),
            'Content-Range': f'bytes 0-{file_size-1}/{file_size}'
        }
        
        return StreamingResponse(
            iterfile(file_path),
            headers=headers,
            media_type="audio/mpeg"
        )
    else:
        return JSONResponse({"error": "Resource not found"}, 404)

@app.get("/api/audio/latest")
async def return_latest_audio_file():
    """
    Returns the latest audio file.

    Returns:
        A JSON response containing the status and the filename of the latest audio file.
    """
    global latest_mp3_response
    if is_file_in_directory(latest_mp3_response, "output"):
        return JSONResponse({"status": "latest", "file": latest_mp3_response})
    else:
        return JSONResponse({"status": "latest", "file": ""}, 404)

async def transcribe(prompt, audio) -> tuple:
    """Send a text message or audio file to an external server
        and get back a text reply from the role-playing chatbot"""
    if audio is not None:
        msg, file_name = await response_from_speech(audio)
        file_path = os.path.join(os.getcwd(), "output", file_name)
        return (msg, file_path)
    elif len(prompt):
        msg, file_name = await response_from_text(prompt)
        file_path = os.path.join(os.getcwd(), "output", file_name)
        return (msg, file_path)
    elif audio is None and prompt is None:
        return "Please provide either a prompt or an audio file"

# The user interface
with gr.Blocks() as demo:
    gr.Markdown("# ChatBot App Demo")
    gr.HTML("""
    <html>
    <body>
        <p>1) Type in the textbox to start, or press the "Record" button to use your voice (remember to press "Stop"!).</p>
        <p>2) Press "Send" to send your message or voice to the chatbot.</p>
        <p>3) Press "Clear" to clear the input textbox and voice recording.</p>
        <p> <u>Note:</u><br> 
            If there's both text and voice recording, the voice recording takes higher priority and is sent.</p>
    </body>
    </html>
    """)
    with gr.Column():    
        with gr.Row():
            with gr.Column():
                output_text = gr.Textbox(lines=2, label="Output")
                output_audio = gr.Audio(label="Chatbot's Voice", type="filepath", format="mp3")

            with gr.Column():
                prompt_input = gr.Textbox(lines=2, label="Type your message", placeholder="Who are you?")
                audio_input = gr.Audio(sources="microphone", label="Use your voice", type="filepath", format="wav")

        with gr.Row():
            clear_button = gr.Button("Clear", variant="secondary")
            clear_button.click(lambda: (None, None), inputs=[], outputs=[prompt_input, audio_input])
            transcribe_button = gr.Button("Send", variant="primary")
            transcribe_button.click(transcribe, inputs=[prompt_input, audio_input], outputs=[output_text, output_audio])

# mount gradio UI 
# Run this from the terminal as you would normally start a FastAPI app: `uvicorn rag_chatbot:app` and
# navigate to http://localhost:8000/gradio in your browser.
app = gr.mount_gradio_app(app, demo, path="/gradio")
