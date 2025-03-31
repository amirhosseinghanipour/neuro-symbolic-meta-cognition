import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from config import config
from utils import TimeUtils

def load_qa_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads question-answering data from a specified JSON file.

    Validates format, requires 'question', 'answer_predicate', 'answer_value'.
    Parses ground truth answer values into appropriate types (datetime/timedelta/int)
    and stores them under 'parsed_answer_value'.

    Args:
        file_path (str): The path to the JSON data file.

    Returns:
        List[Dict[str, Any]]: A list of QA pair dictionaries, or empty list on failure.
    """
    print(f"Attempting to load QA data from: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}"); return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
    except Exception as e: print(f"Error reading/parsing JSON {file_path}: {e}"); return []
    if not isinstance(data, list): print(f"Error: Invalid format in {file_path}. Expected list."); return []

    processed_data = []
    required_keys = {"question", "answer_predicate", "answer_value"}
    for i, item in enumerate(data):
        if not isinstance(item, dict): continue
        if not required_keys.issubset(item.keys()):
            # print(f"Warning: Skipping item {i} due to missing keys: {required_keys - item.keys()}")
            continue

        gt_pred = item['answer_predicate']; gt_val_str = item['answer_value']
        parsed_gt_value = None
        if isinstance(gt_val_str, str):
             if 'time' in gt_pred.lower() or 'when' in gt_pred.lower(): parsed_gt_value = TimeUtils.parse_time(gt_val_str)
             elif 'duration' in gt_pred.lower() or 'long' in gt_pred.lower(): parsed_gt_value = TimeUtils.parse_duration(gt_val_str)
             elif gt_val_str.isdigit():
                  try: parsed_gt_value = int(gt_val_str)
                  except ValueError: pass
        elif isinstance(gt_val_str, (int, float)): # Handle numeric answers directly
             parsed_gt_value = gt_val_str

        item['parsed_answer_value'] = parsed_gt_value
        processed_data.append(item)

    print(f"Successfully loaded and processed {len(processed_data)} QA pairs from {file_path}.")
    return processed_data

def create_sample_data_file(file_path: str = config.DATA_PATH):
    """
    Creates a sample JSON data file if it doesn't exist.

    Args:
        file_path (str): The target path for the sample data file.
    """
    if os.path.exists(file_path): return
    sample_data = [
        {"id": "q1", "question": "If I have a meeting at 3 PM and it takes 30 minutes to get there, when should I leave?", "answer_predicate": "departure_time", "answer_value": "2:30 PM"},
        {"id": "q2", "question": "The workshop starts at 9 AM and runs for 3 hours. What time does it finish?", "answer_predicate": "end_time", "answer_value": "12:00 PM"},
        {"id": "q3", "question": "My flight is at 18:00 and I need 1 hour 15 minutes for travel. What's the latest departure time?", "answer_predicate": "departure_time", "answer_value": "4:45 PM"},
        {"id": "q4", "question": "The project deadline is 5 PM. I need 2 hours to review it. When should I start?", "answer_predicate": "start_work_time", "answer_value": "3:00 PM"},
        {"id": "q5", "question": "How long did the meeting last if it started at 2pm and ended at 4:30 PM?", "answer_predicate": "duration", "answer_value": "2 hours and 30 minutes"},
        {"id": "q6", "question": "My train leaves at 10:00 AM. The journey is 1 hr 15 min long. When do I arrive?", "answer_predicate": "arrival_time", "answer_value": "11:15 AM"}
    ]
    try:
        data_dir = os.path.dirname(file_path)
        if data_dir and not os.path.exists(data_dir): os.makedirs(data_dir); print(f"Created data directory: {data_dir}")
        with open(file_path, 'w', encoding='utf-8') as f: json.dump(sample_data, f, indent=2)
        print(f"Created sample data file: {file_path}")
    except Exception as e: print(f"Error creating sample data file {file_path}: {e}")

if __name__ == '__main__':
    print("Attempting to create sample data file...")
    create_sample_data_file()
    print("Data loader script finished.")
