from openai import OpenAI
import json
from pathlib import Path
import subprocess
import requests
import time
from tqdm import tqdm

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def start_vllm(model_path):
    import sys
    import os 
    env = os.environ.copy()

    vllm_proc = subprocess.Popen([
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--trust_remote_code", "--allowed-local-media-path", "/",
        "--max-model-len", "16384",
        "--media-io-kwargs", json.dumps({"video": {"num_frames": 64}}),
        "--gpu-memory-utilization",  "0.75"
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL, 
    env=env)
    
    url = "http://localhost:8000/v1/models"
    
    totaltime = 600
    sleeptime = 10
    for i in range(totaltime // sleeptime):  # wait up to ~600 seconds
        try:
            r = requests.get(url, timeout=1)
            if r.status_code == 200:
                print("vLLM server is ready with", model_path)
                break
            else:
                print("waiting for vLLM server up", model_path)
        except:
            print("waiting for vLLM server up", model_path)
            
        time.sleep(sleeptime)
    else:
        raise RuntimeError("vLLM server did not start in time")
    
    return vllm_proc


# Video input inference
def run_video(video_url, question, model_path) -> None:
    video_url = str(video_url)
    prompt = "Look at the video, and then fill in the blank accordingly from one of the given choices Only output the correct answer. \n" \
                          f"{question['question']}. \nChoices: {question['choices']}"

    chat_completion_from_url = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": f"file:{video_url}"},
                    },
                    {"type": "text", "text": prompt
                     },
                ],
            }
        ],
        seed=3407,
        temperature=0.1,
        top_p=0.8,
        extra_body={"top_k": 20, "repetition_penalty":1.0, "presence_penalty":1.5, "guided_choice": question['choices']},
        model=model_path,
        max_completion_tokens=16,
        
    )

    result = chat_completion_from_url.choices[0].message.content
    return result

def measure(test, videos, model_path):
    videos = Path(videos)

    data = json.load(open(test))
    total = {}
    correct = {}
    for elem in tqdm(data, desc=f"{model_path}:{videos}"):
        question = elem['metadata']
        video_url = videos / elem['video']
        result = run_video(str(video_url), question, model_path)
        answer = question['answer']
        type = question["type"]
        if result == answer:
            correct[type] = correct.get(type, 0) + 1
        total[type] = total.get(type, 0) + 1
    return correct, total

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', required=True)

    parser.add_argument('--model', required=True)
    
    parser.add_argument('--videos_trajectory', required=True)
    parser.add_argument('--videos_plain', required=True)
    
    args = parser.parse_args()
    results = []
    models = [args.model, 'Qwen/Qwen3-VL-4B-Instruct']
    
    for model_path in models:
        vllm_proc = start_vllm(model_path)
        for video_path in [args.videos_plain, args.videos_trajectory]:
            correct, total = measure(args.test, video_path, model_path)
            result = [correct, total, model_path, video_path]
            results.append(result)
            print(result)

        vllm_proc.terminate()

    print("============================")
    for result in results:
        print(result)
    
    print("============================")