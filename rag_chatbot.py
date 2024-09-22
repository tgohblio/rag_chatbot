import os
import argparse
import requests
import json
from flask import Flask, request, jsonify

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
    """LLM model with RAG"""
    def __init__(self, config: dict):
        self._model = OpenAI("gpt-4o-mini")
        self._system_prompt = config["system prompt"]
        self._init_message = [
            ChatMessage(role="system", content=self._system_prompt),
        ]
        if len(config["folder path"]):
            self._data = SimpleDirectoryReader(input_dir=config["folder path"]).load_data()
            self._index = VectorStoreIndex.from_documents(self._data)
            self._chat_engine = self._index.as_chat_engine(
                chat_mode="react",
                llm=self._model,
                chat_history=self._init_message,
                verbose=True
            )

    def chat_with_rag(self, prompt:str) -> str:
        response = self._chat_engine.chat(prompt)
        return str(response)

    def chat(self) -> str:
        pass

    def clear_message_history(self):
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

try:
    if setup("Sorting Hat") == True:
        # app = Flask(__name__)
        resp = chatbot.chat_with_rag("Summarize the content of the article in the folder")
        print(resp)
    else:
        raise Exception
except Exception as e:
    # Catch any exception and print its content
    print(f"An exception occurred: {e}")



# @app.post("/text")
# def getResponseFromText():
#     data = request.get_json()
#     if data is None:
#         return jsonify({"error": "Invalid JSON input"}), 400
#     prompt = data.get("text")
#     msg = callLLM(prompt)
#     output_path = textToSpeech(msg)
#     return jsonify(content="application/json", text=prompt, file={output_path})


# @app.post("/audio")
# def getResponseFromSpeech():
#     data = request.get_json()
#     if data is None:
#         return jsonify({"error": "Invalid JSON input"}), 400
#     filename = data.get("file")
#     prompt =  speechToText(filename)
#     msg = callLLM(prompt)
#     output_path = textToSpeech(msg)
#     return jsonify(content="application/json", text=prompt, file={output_path})

# if __name__ == "__main__":
#     app.run(debug=True)
