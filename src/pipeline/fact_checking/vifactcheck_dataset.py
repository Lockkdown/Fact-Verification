"""
ViFactCheck Dataset Loader

Load và chuẩn hóa ViFactCheck dataset từ HuggingFace cho evaluation.
Dataset: https://huggingface.co/datasets/tranthaihoa/vifactcheck

Usage:
    from src.pipeline.fact_checking.vifactcheck_dataset import load_vifactcheck
    
    # Load dev set
    dev_data = load_vifactcheck("dev")
    
    # Load test set
    test_data = load_vifactcheck("test")
"""

from datasets import load_dataset
from typing import Dict, List, Literal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Label names for integer labels in ViFactCheck
LABEL_NAMES = {
    0: "Support",   # labels == 0
    1: "Refute",    # labels == 1
    2: "NEI",       # labels == 2 (Not Enough Info)
}


def load_vifactcheck(
    split: Literal["dev", "test"] = "dev",
    return_raw: bool = False
) -> Dict[str, List]:
    """
    Load ViFactCheck dataset từ HuggingFace và chuẩn hóa format.
    
    Args:
        split: "dev" hoặc "test"
        return_raw: Nếu True, trả về raw dataset từ HuggingFace
    
    Returns:
        Dictionary với các key:
            - statements: List[str] - Claim cần verify
            - evidences: List[str] - Evidence text
            - labels: List[int] - Label (0=Support, 1=Refute, 2=NEI)
            - urls: List[str] - Source URL
            - contexts: List[str] - Context (optional)
    
    Raises:
        ValueError: Nếu split không hợp lệ
    """
    if split not in ["dev", "test"]:
        raise ValueError(f"Invalid split: {split}. Must be 'dev' or 'test'.")
    
    logger.info(f"Loading ViFactCheck dataset (split={split})...")
    
    try:
        # Load từ HuggingFace
        dataset = load_dataset("tranthaihoa/vifactcheck")
        
        # Lấy split tương ứng
        split_data = dataset[split]
        
        if return_raw:
            logger.info(f"Loaded {len(split_data)} samples (raw format)")
            return split_data
        
        # Chuẩn hóa format
        processed = {
            "statements": [],
            "evidences": [],
            "labels": [],
            "urls": [],
            "contexts": []
        }
        
        for sample in split_data:
            # Statement (claim) - cột "Statement" trong dataset
            processed["statements"].append(sample["Statement"])
            
            # Evidence text - cột "Evidence"
            processed["evidences"].append(sample["Evidence"])
            
            # Label: ViFactCheck dùng integer labels (0/1/2)
            label_int = int(sample["labels"])
            if label_int not in LABEL_NAMES:
                logger.warning(f"Unknown label id: {label_int}, skipping sample")
                continue
            processed["labels"].append(label_int)
            
            # URL source - cột "Url"
            processed["urls"].append(sample.get("Url", ""))
            
            # Context (optional field) - cột "Context"
            processed["contexts"].append(sample.get("Context", ""))
        
        logger.info(f"Loaded {len(processed['statements'])} samples from {split} set")
        logger.info(f"  - Support: {processed['labels'].count(0)}")
        logger.info(f"  - Refute: {processed['labels'].count(1)}")
        logger.info(f"  - NEI: {processed['labels'].count(2)}")
        
        return processed
    
    except Exception as e:
        logger.error(f"Failed to load ViFactCheck dataset: {e}")
        raise


def load_vifactcheck_subset(
    split: Literal["dev", "test"] = "dev",
    max_samples: int = 100
) -> Dict[str, List]:
    """
    Load subset của ViFactCheck cho testing/debugging nhanh.
    
    Args:
        split: "dev" hoặc "test"
        max_samples: Số lượng samples tối đa
    
    Returns:
        Dictionary với format giống load_vifactcheck()
    """
    logger.info(f"Loading subset (max {max_samples} samples)...")
    
    full_data = load_vifactcheck(split)
    
    # Slice first N samples
    subset = {
        key: values[:max_samples]
        for key, values in full_data.items()
    }
    
    logger.info(f"Loaded subset: {len(subset['statements'])} samples")
    return subset


def get_label_distribution(data: Dict[str, List]) -> Dict[str, int]:
    """
    Tính label distribution từ processed data.
    
    Args:
        data: Output từ load_vifactcheck()
    
    Returns:
        Dictionary: {"Support": count, "Refute": count, "NEI": count}
    """
    labels = data["labels"]
    return {
        "Support": labels.count(0),
        "Refute": labels.count(1),
        "NEI": labels.count(2)
    }


if __name__ == "__main__":
    # Test loading
    print("=" * 60)
    print("Testing ViFactCheck Dataset Loader")
    print("=" * 60)
    
    # Load dev set
    print("\n[1] Loading dev set...")
    dev = load_vifactcheck("dev")
    print(f"  Total samples: {len(dev['statements'])}")
    print(f"  Label distribution: {get_label_distribution(dev)}")
    
    # Sample
    print(f"\n[2] First sample:")
    print(f"  Statement: {dev['statements'][0][:100]}...")
    print(f"  Evidence: {dev['evidences'][0][:100]}...")
    print(f"  Label: {LABEL_NAMES[dev['labels'][0]]}")
    
    # Load test set
    print("\n[3] Loading test set...")
    test = load_vifactcheck("test")
    print(f"  Total samples: {len(test['statements'])}")
    print(f"  Label distribution: {get_label_distribution(test)}")
    
    # Subset test
    print("\n[4] Loading subset (10 samples)...")
    subset = load_vifactcheck_subset("dev", max_samples=10)
    print(f"  Subset size: {len(subset['statements'])}")
    
    print("\n✅ All tests passed!")
