import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# 1. Load environment variables (.env)
load_dotenv()
client = OpenAI()

# 2. Define Pydantic Schemas
class RestockItem(BaseModel):
    item_name: str = Field(description="Name or brand of the product restocked")
    quantity: int = Field(description="Number of units added")
    category: str = Field(description="Category such as Beverage, Snack, or Fresh")

class RestockReport(BaseModel):
    machine_id: str = Field(description="Identifier or location of the vending machine")
    items_added: List[RestockItem]
    needed_next_run: List[str] = Field(description="Items mentioned as low or needed next time")
    notes: Optional[str] = Field(description="Any extra operational or restock notes")

# 3. Transcribe Audio File
def transcribe_audio(file_path: str) -> str:
    print(f"Transcribing {file_path}...")
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model="gpt-transcribe",
            keywords=["Cafination Station", "Celsius", "Doritos", "Snickers"]
        )
    return transcript.text

# 4. Extract Structured Data via Pydantic
def extract_inventory_data(transcript_text: str) -> RestockReport:
    print("Extracting structured inventory data...")
    completion = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an automated logistics and inventory parsing assistant for Cafination Station."
            },
            {
                "role": "user",
                "content": f"Extract the restock details into the required schema: {transcript_text}"
            },
        ],
        response_format=RestockReport,
    )
    return completion.choices[0].message.parsed

# 5. Save Validated Record to Local JSONL File
def save_report(report: RestockReport, log_file: str = "inventory_log.jsonl"):
    record = report.model_dump()
    record["logged_at"] = datetime.now().isoformat()
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"Saved record to {log_file}")

if __name__ == "__main__":
    audio_file = "restock_note.m4a"
    
    if Path(audio_file).exists():
        raw_text = transcribe_audio(audio_file)
        print(f"\nTranscript: \"{raw_text}\"\n")
    else:
        print(f"No '{audio_file}' found—running with sample transcript...\n")
        raw_text = (
            "Hey, serviced machine 4 at the Brighton location. "
            "Added 12 cans of Celsius Peach Vibe, 8 bags of Doritos Nacho Cheese, "
            "and 5 Snickers bars. Need to bring Monster Energy next run. Machine is clean."
        )

    parsed_report = extract_inventory_data(raw_text)
    print("\n--- Validated JSON Output ---")
    print(parsed_report.model_dump_json(indent=2))
    save_report(parsed_report)