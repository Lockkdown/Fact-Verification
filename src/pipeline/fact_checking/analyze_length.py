import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from pathlib import Path
from tqdm import tqdm

def analyze_length(file_path):
    print(f"🔍 Analyzing {file_path}...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    lengths = []
    truncated_count = 0
    total_count = 0
    limit = 512
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            total_count += 1
            item = json.loads(line)
            
            # Giả lập cách chúng ta đang nối chuỗi trong train.py
            # statement [SEP] evidence
            text = item['statement'] + " " + tokenizer.sep_token + " " + item['evidence']
            
            # Tokenize (không truncate để đếm thật)
            tokens = tokenizer.encode(text, add_special_tokens=True)
            length = len(tokens)
            lengths.append(length)
            
            if length > limit:
                truncated_count += 1

    lengths = np.array(lengths)
    
    print("\n" + "="*60)
    print(f"📊 DATA LENGTH STATISTICS ({Path(file_path).name})")
    print("="*60)
    print(f"Total samples:      {total_count}")
    print(f"Truncated (>512):   {truncated_count} ({truncated_count/total_count*100:.2f}%)")
    print(f"Min Length:         {np.min(lengths)}")
    print(f"Max Length:         {np.max(lengths)}")
    print(f"Mean Length:        {np.mean(lengths):.2f}")
    print(f"Median Length:      {np.median(lengths)}")
    print(f"90th Percentile:    {np.percentile(lengths, 90)}")
    print(f"95th Percentile:    {np.percentile(lengths, 95)}")
    print("="*60)

if __name__ == "__main__":
    base_path = Path("dataset/processed/vifactcheck")
    analyze_length(base_path / "vifactcheck_train.jsonl")
    analyze_length(base_path / "vifactcheck_test.jsonl")
