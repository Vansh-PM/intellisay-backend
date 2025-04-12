from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper
import tempfile
import os
import openai
import supabase
from langdetect import detect
from supabase import create_client, Client

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class TaskResponse(BaseModel):
    transcript: str
    task: str
    language: str

# Load Whisper model once
model = whisper.load_model("base")  # Options: tiny, base, small, medium, large

@app.post("/transcribe", response_model=TaskResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp:
        temp.write(await file.read())
        temp.flush()
        audio_path = temp.name

    # Step 1: Transcribe audio locally with Whisper
    result = model.transcribe(audio_path)
    transcript = result["text"]

    # Step 2: Detect language
    language = detect(transcript)

    # Step 3: Use GPT-3.5 to extract a task from transcript
    prompt = f"""
    You are an assistant that extracts todo tasks from a user sentence.
    Sentence: "{transcript}"
    Task (short form):
    """
    gpt_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    task = gpt_response["choices"][0]["message"]["content"].strip()

    # Step 4: Save to Supabase
    supabase_client.table("tasks").insert({
        "transcript": transcript,
        "task": task,
        "language": language
    }).execute()

    return {"transcript": transcript, "task": task, "language": language}
