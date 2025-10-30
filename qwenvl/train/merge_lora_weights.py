
import torch
from peft import PeftModel, PeftConfig
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

def merge_lora(lora_path, save_path):
    peft_cfg = PeftConfig.from_pretrained(lora_path)
    base_model_id = peft_cfg.base_model_name_or_path  
    print(base_model_id)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_id,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
            device_map='cpu'
    )

    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    
    merged = model.merge_and_unload() 

    processor = AutoProcessor.from_pretrained(base_model_id)

    merged.save_pretrained(save_path, safe_serialization=True)
    processor.save_pretrained(save_path)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--save', required=True)
    
    args = parser.parse_args()
    merge_lora(args.model, args.save)
    