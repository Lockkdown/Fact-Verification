"""
Preprocessing ViFactCheck Dataset với pyvi (thay vì VnCoreNLP)
Nhanh hơn, dễ cài đặt, không cần Java
"""
import json
import os
import argparse
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# pyvi for word segmentation
try:
    from pyvi import ViTokenizer
    PYVI_AVAILABLE = True
except ImportError:
    PYVI_AVAILABLE = False
    ViTokenizer = None


# === CẤU HÌNH MẶC ĐỊNH ===
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "../../../dataset/processed/vifactcheck_pyvi"

PHOBERT_MODEL = "vinai/phobert-base"
DEFAULT_MAX_LENGTH = 256


# === LABEL MAPPING ===
LABEL_MAP = {
    0: "SUPPORTS",
    1: "REFUTES",
    2: "NOT ENOUGH INFO"
}

LABEL_STR_TO_INT = {v: k for k, v in LABEL_MAP.items()}


# === PYVI WORD SEGMENTATION ===
def word_segment_text(text: str) -> str:
    """Word segment text using pyvi.
    
    Args:
        text: Raw Vietnamese text
    
    Returns:
        Segmented text (e.g., "Ông Nguyễn_Văn_A đang làm_việc .")
    """
    if not PYVI_AVAILABLE:
        return text
    
    try:
        # pyvi.ViTokenizer.tokenize() trả về string đã segment
        return ViTokenizer.tokenize(text)
    except Exception:
        return text


def normalize_text(text: str, use_pyvi: bool = True) -> str:
    """Normalize text: Unicode normalization + word segmentation + trim whitespace.
    
    Args:
        text: Raw text
        use_pyvi: Whether to use pyvi for word segmentation
    
    Returns:
        Normalized text
    """
    # Normalize Unicode (NFC)
    text = unicodedata.normalize('NFC', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Word segmentation (nếu có pyvi)
    if use_pyvi and PYVI_AVAILABLE:
        text = word_segment_text(text)
    
    return text.strip()


# === ĐỌC VÀ XỬ LÝ DỮ LIỆU ===
def load_vifactcheck_from_hf(split: str, use_pyvi: bool = True) -> Tuple[List[Dict], Dict]:
    """Đọc ViFactCheck từ HuggingFace datasets.
    
    Args:
        split: 'train', 'dev', hoặc 'test'
        use_pyvi: Whether to use pyvi for word segmentation
    
    Returns:
        Tuple of (samples, stats)
    """
    print(f"\n⏳ Loading ViFactCheck {split} split from HuggingFace...")
    
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("⚠️ Please install datasets: pip install datasets")
    
    dataset = load_dataset("tranthaihoa/vifactcheck", split=split)
    
    samples = []
    empty_samples = 0
    total_lines = len(dataset)
    
    for idx, data in enumerate(tqdm(dataset, desc=f"Processing {split}")):
        statement = data.get('Statement', '')
        evidence = data.get('Evidence', '')
        context = data.get('Context', '')
        label = data.get('labels', -1)
        sample_id = data.get('index', idx)
        
        if label not in [0, 1, 2]:
            print(f"⚠️ Invalid label {label} at index {sample_id}, skipping...")
            continue
        
        # Normalize text (Unicode + word segmentation + whitespace)
        statement_norm = normalize_text(statement, use_pyvi)
        evidence_norm = normalize_text(evidence, use_pyvi)
        context_norm = normalize_text(context, use_pyvi)
        
        if not statement_norm:
            empty_samples += 1
            continue
        
        samples.append({
            "statement": statement_norm,
            "evidence": evidence_norm,
            "context": context_norm,
            "label": label,
            "sample_id": str(sample_id)
        })
    
    stats = {
        'total_lines': total_lines,
        'valid_samples': len(samples),
        'empty_samples': empty_samples,
        'skipped_total': empty_samples
    }
    
    return samples, stats


def estimate_max_length(samples: List[Dict], tokenizer, percentile: float = 95.0) -> int:
    """Ước lượng max_length từ dữ liệu."""
    print(f"\n{'='*60}")
    print("ƯỚC LƯỢNG MAX_LENGTH TỪ TRAIN SET")
    print(f"{'='*60}")
    
    lengths = []
    
    import random
    sampled = random.sample(samples, min(5000, len(samples)))
    
    for item in tqdm(sampled, desc="Tokenizing samples"):
        token_ids = tokenizer.encode(
            item['statement'],
            item['evidence'],
            truncation=False,
            add_special_tokens=True
        )
        lengths.append(len(token_ids))
    
    lengths_array = np.array(lengths)
    p50 = np.percentile(lengths_array, 50)
    p75 = np.percentile(lengths_array, 75)
    p95 = np.percentile(lengths_array, 95)
    p99 = np.percentile(lengths_array, 99)
    max_len = np.max(lengths_array)
    
    print(f"\nLength statistics (tokens):")
    print(f"  • P50: {p50:.0f}")
    print(f"  • P75: {p75:.0f}")
    print(f"  • P95: {p95:.0f}")
    print(f"  • P99: {p99:.0f}")
    print(f"  • Max: {max_len:.0f}")
    
    estimated = int(np.percentile(lengths_array, percentile))
    
    if estimated < 128:
        estimated = 128
    elif estimated > 384:
        estimated = 384
    
    print(f"\n✓ Estimated max_length (P{percentile:.0f}): {estimated}")
    print(f"  → Recommend: 256")
    
    return estimated


def save_jsonl(samples: List[Dict], output_path: str):
    """Lưu samples ra JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✓ Saved {len(samples)} samples to {output_path}")


def print_statistics(samples: List[Dict], stats: Dict, split_name: str):
    """In thống kê dataset."""
    labels = [s['label'] for s in samples]
    label_counts = Counter(labels)
    
    print(f"\n{split_name.upper()} Statistics:")
    print(f"  • Total lines: {stats['total_lines']}")
    print(f"  • Valid samples: {stats['valid_samples']}")
    print(f"  • Skipped total: {stats['skipped_total']}")
    print(f"    - Empty samples: {stats['empty_samples']}")
    
    if len(samples) > 0:
        print(f"\n  Label distribution:")
        print(f"  • SUPPORTS (0): {label_counts[0]} ({label_counts[0]/len(samples)*100:.1f}%)")
        print(f"  • REFUTES (1): {label_counts[1]} ({label_counts[1]/len(samples)*100:.1f}%)")
        print(f"  • NOT ENOUGH INFO (2): {label_counts[2]} ({label_counts[2]/len(samples)*100:.1f}%)")
        
        imbalance_ratio = max(label_counts.values()) / min(label_counts.values())
        if imbalance_ratio < 1.2:
            print(f"  ✅ Dataset BALANCED (imbalance ratio: {imbalance_ratio:.2f})")
        else:
            print(f"  ⚠ Dataset IMBALANCED (imbalance ratio: {imbalance_ratio:.2f})")
        
        statement_lengths = [len(s['statement'].split()) for s in samples]
        evidence_lengths = [len(s['evidence'].split()) for s in samples]
        context_lengths = [len(s['context'].split()) for s in samples]
        
        print(f"\n  Text length (words):")
        print(f"  • Statement (Claim): avg={np.mean(statement_lengths):.1f}, max={np.max(statement_lengths)}")
        print(f"  • Evidence (Gold):   avg={np.mean(evidence_lengths):.1f}, max={np.max(evidence_lengths)}")
        print(f"  • Context (Full):    avg={np.mean(context_lengths):.1f}, max={np.max(context_lengths)}")


def main():
    import sys
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    parser = argparse.ArgumentParser(description='Preprocessing ViFactCheck Dataset with pyvi')
    
    parser.add_argument('--outdir', type=str, default=str(OUTPUT_DIR),
                       help='Output directory')
    parser.add_argument('--max-length', type=int, default=DEFAULT_MAX_LENGTH,
                       help='Max sequence length (default: 256)')
    parser.add_argument('--no-pyvi', action='store_true',
                       help='Disable pyvi word segmentation')
    
    args = parser.parse_args()
    
    use_pyvi = not args.no_pyvi
    
    print(f"\n{'='*60}")
    print("PREPROCESSING VIFACTCHECK DATASET (pyvi)")
    print(f"{'='*60}")
    print(f"PhoBERT model: {PHOBERT_MODEL}")
    print(f"Max length (default): {args.max_length} tokens")
    print(f"Use pyvi: {use_pyvi}")
    print(f"✅ 3 LABELS: SUPPORTS (0), REFUTES (1), NOT ENOUGH INFO (2)")
    
    # Check pyvi
    if use_pyvi:
        if not PYVI_AVAILABLE:
            print("\n⚠️ pyvi not installed! Install: pip install pyvi")
            print("   Continuing WITHOUT word segmentation...")
            use_pyvi = False
        else:
            print(f"\n✅ pyvi available")
            # Test
            test_text = "Ông Nguyễn Văn A đang làm việc."
            test_result = word_segment_text(test_text)
            print(f"\n📝 Test segmentation:")
            print(f"   Input:  {test_text}")
            print(f"   Output: {test_result}")
    else:
        print(f"\n⚠️ pyvi word segmentation DISABLED")
    
    # Load tokenizer
    print(f"\n{'='*60}")
    print("LOADING TOKENIZER")
    print(f"{'='*60}")
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL)
    print(f"✓ Loaded PhoBERT tokenizer")
    print(f"  • Vocab size: {tokenizer.vocab_size}")
    print(f"  • SEP token: {tokenizer.sep_token}")
    
    # Process train set
    print(f"\n{'='*60}")
    print("PROCESSING TRAIN SET")
    print(f"{'='*60}")
    train_samples, train_stats = load_vifactcheck_from_hf('train', use_pyvi)
    print_statistics(train_samples, train_stats, "train")
    
    # Estimate max_length
    estimated_max_length = estimate_max_length(train_samples, tokenizer, percentile=95.0)
    
    if args.max_length == DEFAULT_MAX_LENGTH:
        final_max_length = estimated_max_length
        print(f"\n✅ Sử dụng ESTIMATED max_length: {final_max_length}")
    else:
        final_max_length = args.max_length
        if final_max_length != estimated_max_length:
            print(f"\n⚠️  User specified max_length={final_max_length}, estimated={estimated_max_length}")
        else:
            print(f"\n✅ User max_length khớp với estimated: {final_max_length}")
    
    # Process dev set
    print(f"\n{'='*60}")
    print("PROCESSING DEV SET")
    print(f"{'='*60}")
    dev_samples, dev_stats = load_vifactcheck_from_hf('dev', use_pyvi)
    print_statistics(dev_samples, dev_stats, "dev")
    
    # Process test set
    print(f"\n{'='*60}")
    print("PROCESSING TEST SET")
    print(f"{'='*60}")
    test_samples, test_stats = load_vifactcheck_from_hf('test', use_pyvi)
    print_statistics(test_samples, test_stats, "test")
    
    # Save to JSONL
    print(f"\n{'='*60}")
    print(f"SAVING TO JSONL")
    print(f"{'='*60}")
    
    outdir = Path(args.outdir)
    save_jsonl(train_samples, str(outdir / "vifactcheck_train.jsonl"))
    save_jsonl(dev_samples, str(outdir / "vifactcheck_dev.jsonl"))
    save_jsonl(test_samples, str(outdir / "vifactcheck_test.jsonl"))
    
    # Final summary
    print(f"\n{'='*60}")
    print("✅ PREPROCESSING COMPLETED (pyvi)!")
    print(f"{'='*60}")
    print(f"Output directory: {outdir.absolute()}")
    print(f"  • vifactcheck_train.jsonl: {len(train_samples)} samples")
    print(f"  • vifactcheck_dev.jsonl: {len(dev_samples)} samples")
    print(f"  • vifactcheck_test.jsonl: {len(test_samples)} samples")
    
    print(f"\n⚡ RECOMMENDED max_length: {final_max_length} tokens (P95)")
    print(f"Format: {{\"statement\": \"...\", \"evidence\": \"...\", \"label\": 0|1|2, \"sample_id\": \"...\"}}")
    print(f"\nLabel mapping:")
    print(f"  • 0 = SUPPORTS")
    print(f"  • 1 = REFUTES")
    print(f"  • 2 = NOT ENOUGH INFO")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
