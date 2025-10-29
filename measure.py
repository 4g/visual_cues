from openai import OpenAI
import json
from pathlib import Path
import subprocess
import requests
import time

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
    ], env=env)
    
    url = "http://localhost:8000/v1/models"
    
    for i in range(600):  # wait up to ~600 seconds
        try:
            r = requests.get(url, timeout=1)
            if r.status_code == 200:
                print("vLLM server is ready with", model_path)
                break
            else:
                print("waiting for vLLM server up", model_path)
        except Exception:
            pass
        time.sleep(1)
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
        temperature=0.0,
        model=model_path,
        max_completion_tokens=16,
        extra_body={"guided_choice": question['choices']},
    )

    result = chat_completion_from_url.choices[0].message.content
    return result

def measure(test, videos, model_path):
    videos = Path(videos)

    data = json.load(open(test))
    total = 0
    correct = 0
    for elem in data[:1000]:
        question = elem['metadata']
        video_url = videos / elem['video']
        result = run_video(str(video_url), question, model_path)
        answer = question['answer']
        if result == answer:
            correct += 1
        total += 1
    return correct, total

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', required=True)

    parser.add_argument('--model_plain', required=True)
    parser.add_argument('--model_trajectory', required=True)
    
    parser.add_argument('--videos_trajectory', required=True)
    parser.add_argument('--videos_plain', required=True)
    
    args = parser.parse_args()
    results = []
    for model_path in [args.model_plain, args.model_trajectory, 'Qwen/Qwen3-VL-2B-Instruct']:
        vllm_proc = start_vllm(model_path)
        for video_path in [args.videos_plain, args.videos_trajectory]:
            correct, total = measure(args.test, video_path, model_path)
            results.append([correct, total, model_path, video_path])
        
        vllm_proc.terminate()
    
    print("============================")
    for result in results:
        print(result)
    
    print("============================")