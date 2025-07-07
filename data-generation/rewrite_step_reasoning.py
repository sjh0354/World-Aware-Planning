import base64
import json
import requests
import copy
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_session = requests.Session()

# HTTP retry
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
_session.mount('http://', HTTPAdapter(max_retries=retries))
_session.mount('https://', HTTPAdapter(max_retries=retries))

def image_to_base64(image_path):
    """img2base64"""
    
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
            # _image_cache[image_path] = encoded
            return encoded
    except Exception as e:
        print(f"img faliure: {image_path} - {str(e)}")
        return ""

def generate_reasoning(user_instruction, previous_steps, current_image, original_action):
    image_base64 = image_to_base64(current_image)
    if not image_base64:
        return ""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text":  f"""
You are a robot operating in a home. Given a task, you must accomplish the task using a defined set of actions to achieve the desired outcome.

User Instruction:
{user_instruction}

Previous Steps:
{previous_steps}

You need to describe the current visual state from the image, output your reasoning steps, and plan. You have decided your next step and your task is to generate the reasoning for the next step. The reasoning should logically explain why the next action is appropriate given the context. Please make your reasoning natural and concise.

Here is an example:

Since I have the remote control in my hand, I need to locate the floor lamp switch to turn on the light. It's likely that the floor lamp switch is nearby or that the remote control has a feature to turn on the light directly. I will first try to find the floor lamp switch.\n

Your next step's action: {original_action}

Enhanced data:
"""
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                }
            ]
        }
    ]

    try:
        response = _session.post(
            "http://0.0.0.0:8000/v1/chat/completions",
            headers={"Content-Type": "application/json", "Authorization": "Bearer sk-111"},
            json={
                "model": "qwen2vl",
                "messages": messages,
                "max_tokens": 300,
                "temperature": 0.7,
                "top_p": 0.9
            },
            # timeout=30
        )
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"API failure: {str(e)}")
        return ""

def process_single_entry(entry):
    """single process"""
    try:
        modified_entry = copy.deepcopy(entry)
        conversations = modified_entry["conversations"]
        images = modified_entry["images"]
        
        previous_steps = []
        image_index = 0
        
        for i, msg in enumerate(conversations):
            if msg["from"] == "gpt" and "Reasoning:" in msg["value"]:
                if image_index >= len(images):
                    break
                
                original_action = msg["value"].split("Action: ")[-1].strip()
                current_image = images[image_index]
                image_index += 1
                
                new_reasoning = generate_reasoning(
                    conversations[0]["value"],
                    "\n".join(previous_steps),
                    current_image,
                    original_action
                )
                
                new_reasoning = new_reasoning.replace("Reasoning:", "").split("Action:")[0].strip()
                
                msg["value"] = f"Reasoning: {new_reasoning}\nAction: {original_action}"
                previous_steps.append(f"Reasoning: {new_reasoning} Action: {original_action}")
        
        return modified_entry
    except Exception as e:
        print(f"Failure: {str(e)}")
        return None

def process_dataset(subset_name):
    json_list = ["path/to/data"]

    for path in json_list:
        try:
            with open(path) as f:
                dataset = json.load(f)
        except Exception as e:
            print(f"file loading failure: {path} - {str(e)}")
            continue

        enhanced_data = []
        with ThreadPoolExecutor(max_workers=20) as executor:  
            futures = [executor.submit(process_single_entry, entry) for entry in dataset]
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    enhanced_data.append(result)
                    if len(enhanced_data) % 10 == 0: 
                        print(f"Processed {len(enhanced_data)}/{len(dataset)} data")

        # 保存结果
        output_path = f"path/to/data"
        with open(output_path, "w") as f:
            json.dump(enhanced_data, f, indent=2)
        print(f"Saved to {output_path}")

def main():
    """argparse"""
    parser = argparse.ArgumentParser(description="tool")
    parser.add_argument("--subset", "-s", required=True, help="subset name")
    args = parser.parse_args()
    
    print(f"Processing {args.subset} subset")
    process_dataset(args.subset)
    print("Finish")

if __name__ == "__main__":
    main()