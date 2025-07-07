import json
import time
import base64
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def create_retry_session(retries=5, backoff_factor=1):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=frozenset(['POST'])
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def enhance_instruction(data_entry, max_retries=3, retry_delay=1):
    human_instruction = data_entry["conversations"][0]["value"]
    image_url = data_entry["images"][-1]

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""Enhance the following instruction into a more elaborate version by adding contextual details and symbolic substitutions. Replace objects and locations with contextual references (e.g., pronouns or implied terms) and include irrelevant but plausible background information. Keep sentences concise and avoid adding new actions.

Examples:

1. Original: "Put two spray bottles in the cabinet"
Enhanced: "The spray bottles are on the shelf, already cleaned from yesterday's use. Move them to the cabinet for storage."

2. Original: "Put a knife in a container"
Enhanced: "That knife on the counter just finished slicing vegetables. Place it in the container to keep the edge protected."

3. Original: "Put washed lettuce in the refrigerator"
Enhanced: "There's a lettuce in the sink—we've prepped enough for dinner. Wash it and store there to keep it fresh."

Now enhance:
Original: {human_instruction}
Enhanced: """
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_to_base64(image_url)}"
                    }
                }
            ]
        }
    ]

    # 使用带重试的Session
    session = create_retry_session()
    
    for attempt in range(max_retries + 1):
        try:
            response = session.post(
                "http://0.0.0.0:8000/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer sk-111"
                },
                json={
                    "model": "qwen2vl",
                    "messages": messages,
                    "max_tokens": 300,
                    "temperature": 0.7,
                    "top_p": 0.9
                },
                timeout=30
            )
            response.raise_for_status()  
            return response.json()["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            if attempt < max_retries:
                wait_time = retry_delay * (2 ** attempt)  
                print(f"Attempt {attempt+1}/{max_retries} fails: retrying after {str(e)}，{wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise Exception(f"Failing after {max_retries} retries: {str(e)}")

def process_entry(entry):
    try:
        enhanced = enhance_instruction(entry)
        return {
            "original": entry["conversations"][0]["value"],
            "enhanced": enhanced,
            "image": entry["images"][-1]
        }
    except Exception as e:
        print(f"Final failure: {str(e)}")
        return None

json_list = ["path/to/data"]

def process_entry(entry):
    try:
        enhanced = enhance_instruction(entry)
        return {
            "original": entry["conversations"][0]["value"],
            "enhanced": enhanced,
            "image": entry["images"][-1]
        }
    except Exception as e:
        print(f"Failure: {str(e)}")
        return None

for path in json_list:
    name = path.split("/")[-1].split(".")[0]
    with open(path) as f:
        dataset = json.load(f)

    enhanced_data = []
    failed_count = 0

    with ThreadPoolExecutor(max_workers=100) as executor: 
        futures = [executor.submit(process_entry, entry) for entry in dataset]
        
        progress = tqdm(total=len(futures), desc=f"Processing {name}", ncols=100)
        for future in futures:
            result = future.result()
            if result:
                enhanced_data.append(result)
            else:
                failed_count += 1
            progress.update()
        progress.close()

    print(f"Success {len(enhanced_data)} ,Failure {failed_count} ")

    save_path = f"path/to/save"

    with open(save_path, "w") as f:
        json.dump(enhanced_data, f, indent=2)
    print(f"Saving to {save_path}")



    