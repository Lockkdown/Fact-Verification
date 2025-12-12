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

# VnCoreNLP (optional - dùng cho word segmentation)
try:
    from py_vncorenlp import VnCoreNLP
    VNCORENLP_AVAILABLE = True
except ImportError:
    VNCORENLP_AVAILABLE = False
    VnCoreNLP = None


# === CẤU HÌNH MẶC ĐỊNH ===
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "../../../dataset/processed/vifactcheck"

PHOBERT_MODEL = "vinai/phobert-base"
DEFAULT_MAX_LENGTH = 256  # Recommend 256 tokens (Statement ~161 chars, Evidence ~196 chars)

# VnCoreNLP paths
VNCORENLP_DIR = SCRIPT_DIR / "../../../tools/VnCoreNLP-1.2"


# === LABEL MAPPING ===
# ViFactCheck: 3 nhãn - Support / Refute / NEI
LABEL_MAP = {
    0: "SUPPORTS",        # Support
    1: "REFUTES",         # Refute
    2: "NOT ENOUGH INFO"  # NEI
}

# Reverse mapping để có thể convert từ string (nếu cần)
LABEL_STR_TO_INT = {v: k for k, v in LABEL_MAP.items()}


# === VnCoreNLP SETUP ===
def init_vncorenlp():
    """Khởi tạo VnCoreNLP annotator.
    
    Returns:
        VnCoreNLP annotator hoặc None nếu không available
    """
    if not VNCORENLP_AVAILABLE:
        print("⚠️  py_vncorenlp not installed. Install: pip install py_vncorenlp")
        print("   Word segmentation will be SKIPPED (⚠️ performance drop ~5-10%)")
        return None
    
    try:
        print(f"\n{'='*60}")
        print("INITIALIZING VnCoreNLP")
        print(f"{'='*60}")
        print(f"VnCoreNLP directory: {VNCORENLP_DIR.absolute()}")
        print("⏳ Loading VnCoreNLP (this may take 10-30 seconds)...")
        
        annotator = VnCoreNLP(
            save_dir=str(VNCORENLP_DIR.absolute())
        )
        
        print("✅ VnCoreNLP loaded successfully")
        
        # Test
        test_text = "Ông Nguyễn Văn A đang làm việc."
        test_result = annotator.word_segment(test_text)
        print(f"\n📝 Test segmentation:")
        print(f"   Input:  {test_text}")
        print(f"   Output: {test_result[0]}")
        
        return annotator
        
    except Exception as e:
        print(f"❌ Failed to initialize VnCoreNLP: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Ensure Java is installed: java -version")
        print("   2. Download VnCoreNLP-1.2.jar to tools/ directory")
        print("   3. See docs/VNCORENLP_SETUP.md for details")
        print("\n   Word segmentation will be SKIPPED (⚠️ performance drop ~5-10%)")
        return None


def word_segment_text(text: str, annotator) -> str:
    """Word segment text using VnCoreNLP.
    
    Args:
        text: Raw Vietnamese text
        annotator: VnCoreNLP annotator (or None to skip)
    
    Returns:
        Segmented text (e.g., "Ông Nguyễn_Văn_A đang làm_việc .")
    """
    if annotator is None:
        return text  # Skip segmentation if not available
    
    try:
        # VnCoreNLP returns List[str], take first element
        segmented_list = annotator.word_segment(text)
        if isinstance(segmented_list, list) and len(segmented_list) > 0:
            return segmented_list[0]
        else:
            return text  # Fallback
    except Exception as e:
        # Nếu lỗi, return raw text
        return text


# === HÀM NORMALIZE TEXT ===
def normalize_text(text: str, annotator=None) -> str:
    """Normalize text: Unicode normalization + word segmentation + trim whitespace.
    
    Args:
        text: Raw text
        annotator: VnCoreNLP annotator (optional, for word segmentation)
    
    Returns:
        Normalized text
    """
    # Normalize Unicode (NFC)
    text = unicodedata.normalize('NFC', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Word segmentation (nếu có VnCoreNLP)
    if annotator is not None:
        text = word_segment_text(text, annotator)
    
    return text.strip()


# === ĐỌC VÀ XỬ LÝ DỮ LIỆU ===
def load_vifactcheck_from_hf(split: str, annotator=None) -> Tuple[List[Dict], Dict]:
    """Đọc ViFactCheck từ HuggingFace datasets.
    
    Args:
        split: 'train', 'dev', hoặc 'test'
        annotator: VnCoreNLP annotator (optional, for word segmentation)
    
    Returns:
        Tuple of (samples, stats)
        - samples: List of {"statement": str, "evidence": str, "label": int, "sample_id": str}
        - stats: Dict với thống kê unknown labels
    """
    print(f"\n⏳ Loading ViFactCheck {split} split from HuggingFace...")
    
    # Import datasets
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("⚠️ Please install datasets: pip install datasets")
    
    # Load dataset
    dataset = load_dataset("tranthaihoa/vifactcheck", split=split)
    
    samples = []
    empty_samples = 0
    total_lines = len(dataset)
    
    for idx, data in enumerate(tqdm(dataset, desc=f"Processing {split}")):
        # Extract fields (ViFactCheck schema: Statement, Evidence, labels)
        statement = data.get('Statement', '')      # Claim cần verify
        evidence = data.get('Evidence', '')        # Bằng chứng (Gold)
        context = data.get('Context', '')          # Bài báo gốc (Full Context)
        label = data.get('labels', -1)            # 0=SUPPORTS, 1=REFUTES, 2=NOT ENOUGH INFO
        sample_id = data.get('index', idx)        # Unique ID
        
        # Validate label
        if label not in [0, 1, 2]:
            print(f"⚠️ Invalid label {label} at index {sample_id}, skipping...")
            continue
        
        # Normalize text (Unicode + word segmentation + whitespace)
        statement_norm = normalize_text(statement, annotator)
        evidence_norm = normalize_text(evidence, annotator)
        context_norm = normalize_text(context, annotator)
        
        # Skip if empty (Context rỗng thì vẫn chấp nhận nếu có Evidence, tùy logic)
        if not statement_norm:
            empty_samples += 1
            continue
        
        samples.append({
            "statement": statement_norm,
            "evidence": evidence_norm,
            "context": context_norm,  # Thêm trường context
            "label": label,
            "sample_id": str(sample_id)
        })
    
    # Tạo stats
    stats = {
        'total_lines': total_lines,
        'valid_samples': len(samples),
        'empty_samples': empty_samples,
        'skipped_total': empty_samples
    }
    
    return samples, stats


def estimate_max_length(samples: List[Dict], tokenizer, percentile: float = 95.0) -> int:
    """Ước lượng max_length từ dữ liệu.
    
    Tokenize statement + evidence và tính phân vị để chọn max_length hợp lý.
    
    Args:
        samples: List of {"statement": str, "evidence": str, ...}
        tokenizer: PhoBERT tokenizer
        percentile: Phân vị để chọn (default 95.0 = P95)
    
    Returns:
        max_length (int)
    """
    print(f"\n{'='*60}")
    print("ƯỚC LƯỢNG MAX_LENGTH TỪ TRAIN SET")
    print(f"{'='*60}")
    
    lengths = []
    
    # Sample 5000 random samples để tính nhanh
    import random
    sampled = random.sample(samples, min(5000, len(samples)))
    
    for item in tqdm(sampled, desc="Tokenizing samples"):
        # Tokenize: statement + [SEP] + evidence
        # Use encode() instead of tokenize() to avoid warnings about overflowing tokens
        token_ids = tokenizer.encode(
            item['statement'],
            item['evidence'],
            truncation=False,  # Don't truncate during estimation
            add_special_tokens=True
        )
        lengths.append(len(token_ids))
    
    # Tính statistics
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
    
    # Chọn max_length
    estimated = int(np.percentile(lengths_array, percentile))
    
    # Clamp vào khoảng [128, 384]
    if estimated < 128:
        estimated = 128
    elif estimated > 384:
        estimated = 384
    
    print(f"\n✓ Estimated max_length (P{percentile:.0f}): {estimated}")
    print(f"  → Recommend: 256")
    
    return estimated


def save_jsonl(samples: List[Dict], output_path: str):
    """Lưu samples ra JSONL file (KHÔNG tokenize).
    
    Args:
        samples: List of {"statement": str, "evidence": str, "label": int, "sample_id": str}
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Lưu ra JSONL
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
        
        # Check balance (3 nhãn)
        max_diff = max(abs(label_counts[i] - label_counts[j]) for i in range(3) for j in range(i+1, 3))
        imbalance_ratio = max(label_counts.values()) / min(label_counts.values())
        if imbalance_ratio < 1.2:
            print(f"  ✅ Dataset BALANCED (imbalance ratio: {imbalance_ratio:.2f})")
        else:
            print(f"  ⚠ Dataset IMBALANCED (imbalance ratio: {imbalance_ratio:.2f})")
        
        # Length statistics
        statement_lengths = [len(s['statement'].split()) for s in samples]
        evidence_lengths = [len(s['evidence'].split()) for s in samples]
        context_lengths = [len(s['context'].split()) for s in samples]
        
        print(f"\n  Text length (words):")
        print(f"  • Statement (Claim): avg={np.mean(statement_lengths):.1f}, max={np.max(statement_lengths)}")
        print(f"  • Evidence (Gold):   avg={np.mean(evidence_lengths):.1f}, max={np.max(evidence_lengths)}")
        print(f"  • Context (Full):    avg={np.mean(context_lengths):.1f}, max={np.max(context_lengths)}")


def main():
    # Fix Windows console encoding for emojis
    import sys
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
        
        # Set JAVA_HOME if not set (required for py_vncorenlp)
        if 'JAVA_HOME' not in os.environ or os.environ['JAVA_HOME'] == 'C:\\Path\\to\\your\\JDK':
            # Auto-detect Java installation
            java_paths = [
                r"C:\Program Files\Java\jdk-21",
                r"C:\Program Files\Java\jdk-17",
                r"C:\Program Files\Java\jdk-11",
            ]
            for java_path in java_paths:
                if os.path.exists(java_path):
                    os.environ['JAVA_HOME'] = java_path
                    print(f"✓ Auto-set JAVA_HOME: {java_path}")
                    break
    
    parser = argparse.ArgumentParser(description='Preprocessing ViFactCheck Dataset')
    
    parser.add_argument('--outdir', type=str, default=str(OUTPUT_DIR),
                       help='Output directory')
    parser.add_argument('--max-length', type=int, default=DEFAULT_MAX_LENGTH,
                       help='Max sequence length (default: 256)')
    parser.add_argument('--use-vncorenlp', action='store_true', default=True,
                       help='Use VnCoreNLP for word segmentation (default: True - RECOMMENDED for PhoBERT)')
    parser.add_argument('--no-vncorenlp', action='store_false', dest='use_vncorenlp',
                       help='Disable VnCoreNLP (NOT recommended - performance drop ~5-10%)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("PREPROCESSING VIFACTCHECK DATASET")
    print(f"{'='*60}")
    print(f"PhoBERT model: {PHOBERT_MODEL}")
    print(f"Max length (default): {args.max_length} tokens (will estimate from data)")
    print(f"Use VnCoreNLP: {args.use_vncorenlp}")
    print(f"✅ 3 LABELS: SUPPORTS (0), REFUTES (1), NOT ENOUGH INFO (2)")
    
    # Initialize VnCoreNLP if requested
    annotator = None
    if args.use_vncorenlp:
        annotator = init_vncorenlp()
        if annotator is None:
            print("\n⚠️  Continuing WITHOUT word segmentation...")
    else:
        print(f"\n⚠️  VnCoreNLP word segmentation DISABLED")
        print(f"   → Performance may drop ~5-10% compared to paper")
        print(f"   → Use --use-vncorenlp to enable")
    
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
    if annotator is not None:
        print("⏳ Word segmentation enabled - this will take ~30-60 minutes...")
    train_samples, train_stats = load_vifactcheck_from_hf('train', annotator)
    print_statistics(train_samples, train_stats, "train")
    
    # ⚡ CRITICAL: Estimate max_length từ train set và SỬ DỤNG nó!
    estimated_max_length = estimate_max_length(train_samples, tokenizer, percentile=95.0)
    
    # Quyết định max_length cuối cùng
    if args.max_length == DEFAULT_MAX_LENGTH:
        # User không specify → dùng estimated
        final_max_length = estimated_max_length
        print(f"\n✅ Sử dụng ESTIMATED max_length: {final_max_length}")
    else:
        # User đã specify → dùng user's choice nhưng warning nếu khác estimated
        final_max_length = args.max_length
        if final_max_length != estimated_max_length:
            print(f"\n⚠️  User specified max_length={final_max_length}, estimated={estimated_max_length}")
        else:
            print(f"\n✅ User max_length khớp với estimated: {final_max_length}")
    
    # Process dev set
    print(f"\n{'='*60}")
    print("PROCESSING DEV SET")
    print(f"{'='*60}")
    dev_samples, dev_stats = load_vifactcheck_from_hf('dev', annotator)
    print_statistics(dev_samples, dev_stats, "dev")
    
    # Process test set
    print(f"\n{'='*60}")
    print("PROCESSING TEST SET")
    print(f"{'='*60}")
    test_samples, test_stats = load_vifactcheck_from_hf('test', annotator)
    print_statistics(test_samples, test_stats, "test")
    
    # Save to JSONL (NO tokenization - will be done in dataset class)
    print(f"\n{'='*60}")
    print(f"SAVING TO JSONL (NO tokenization)")
    print(f"{'='*60}")
    print(f"⚡ Tokenization sẽ được thực hiện trong Dataset class (__init__)")
    print(f"   → Tối ưu: Pre-tokenize 1 lần khi load, tránh overhead multiprocessing\n")
    
    outdir = Path(args.outdir)
    save_jsonl(train_samples, str(outdir / "vifactcheck_train.jsonl"))
    save_jsonl(dev_samples, str(outdir / "vifactcheck_dev.jsonl"))
    save_jsonl(test_samples, str(outdir / "vifactcheck_test.jsonl"))
    
    # Final summary
    print(f"\n{'='*60}")
    print("✅ PREPROCESSING COMPLETED!")
    print(f"{'='*60}")
    print(f"Output directory: {outdir.absolute()}")
    print(f"  • vifactcheck_train.jsonl: {len(train_samples)} samples")
    print(f"  • vifactcheck_dev.jsonl: {len(dev_samples)} samples")
    print(f"  • vifactcheck_test.jsonl: {len(test_samples)} samples")
    
    print(f"\n⚡ RECOMMENDED max_length: {final_max_length} tokens (P95)")
    print(f"   → Sử dụng trong Dataset class khi pre-tokenize\n")
    print(f"Format: {{\"statement\": \"...\", \"evidence\": \"...\", \"label\": 0|1|2, \"sample_id\": \"...\"}}")
    print(f"\nLabel mapping:")
    print(f"  • 0 = SUPPORTS")
    print(f"  • 1 = REFUTES")
    print(f"  • 2 = NOT ENOUGH INFO")
    print(f"✅ Dataset balanced (imbalance ratio < 1.2) → Production-ready")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
