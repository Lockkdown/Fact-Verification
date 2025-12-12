"""
ViFactCheck Dataset Analysis

Phân tích và generate report cho ViFactCheck dataset.
Output: docs/VIFACTCHECK_ANALYSIS.md

Usage:
    python src/pipeline/fact_checking/analyze_vifactcheck.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.fact_checking.vifactcheck_dataset import (
    load_vifactcheck,
    get_label_distribution,
    LABEL_NAMES
)
import logging
from collections import Counter
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_statistics(data: dict) -> dict:
    """Tính các thống kê cơ bản cho dataset."""
    statements = data["statements"]
    evidences = data["evidences"]
    
    stats = {
        "total_samples": len(statements),
        "label_distribution": get_label_distribution(data),
        "statement_lengths": {
            "mean": np.mean([len(s.split()) for s in statements]),
            "median": np.median([len(s.split()) for s in statements]),
            "min": min([len(s.split()) for s in statements]),
            "max": max([len(s.split()) for s in statements])
        },
        "evidence_lengths": {
            "mean": np.mean([len(e.split()) for e in evidences]),
            "median": np.median([len(e.split()) for e in evidences]),
            "min": min([len(e.split()) for e in evidences]),
            "max": max([len(e.split()) for e in evidences])
        }
    }
    
    return stats


def generate_report(dev_stats: dict, test_stats: dict, output_path: str):
    """Generate markdown report."""
    
    report = f"""# ViFactCheck Dataset Analysis

**Generated:** {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Source:** https://huggingface.co/datasets/tranthaihoa/vifactcheck  
**License:** MIT

---

## 📊 Dataset Overview

### Size
- **Dev set:** {dev_stats['total_samples']} samples
- **Test set:** {test_stats['total_samples']} samples
- **Total:** {dev_stats['total_samples'] + test_stats['total_samples']} samples

---

## 🏷️ Label Distribution

### Dev Set
- **Support:** {dev_stats['label_distribution']['Support']} ({dev_stats['label_distribution']['Support'] / dev_stats['total_samples'] * 100:.1f}%)
- **Refute:** {dev_stats['label_distribution']['Refute']} ({dev_stats['label_distribution']['Refute'] / dev_stats['total_samples'] * 100:.1f}%)
- **NEI:** {dev_stats['label_distribution']['NEI']} ({dev_stats['label_distribution']['NEI'] / dev_stats['total_samples'] * 100:.1f}%)

### Test Set
- **Support:** {test_stats['label_distribution']['Support']} ({test_stats['label_distribution']['Support'] / test_stats['total_samples'] * 100:.1f}%)
- **Refute:** {test_stats['label_distribution']['Refute']} ({test_stats['label_distribution']['Refute'] / test_stats['total_samples'] * 100:.1f}%)
- **NEI:** {test_stats['label_distribution']['NEI']} ({test_stats['label_distribution']['NEI'] / test_stats['total_samples'] * 100:.1f}%)

**Observation:**
- Dataset is {'balanced' if max(dev_stats['label_distribution'].values()) / min(dev_stats['label_distribution'].values()) < 1.5 else 'imbalanced'}
- NEI class is {'dominant' if dev_stats['label_distribution']['NEI'] > dev_stats['total_samples'] * 0.4 else 'normal'}

---

## 📏 Text Length Statistics

### Statement (Claim) Length (words)

| Split | Mean | Median | Min | Max |
|-------|------|--------|-----|-----|
| Dev | {dev_stats['statement_lengths']['mean']:.1f} | {dev_stats['statement_lengths']['median']:.1f} | {dev_stats['statement_lengths']['min']} | {dev_stats['statement_lengths']['max']} |
| Test | {test_stats['statement_lengths']['mean']:.1f} | {test_stats['statement_lengths']['median']:.1f} | {test_stats['statement_lengths']['min']} | {test_stats['statement_lengths']['max']} |

### Evidence Length (words)

| Split | Mean | Median | Min | Max |
|-------|------|--------|-----|-----|
| Dev | {dev_stats['evidence_lengths']['mean']:.1f} | {dev_stats['evidence_lengths']['median']:.1f} | {dev_stats['evidence_lengths']['min']} | {dev_stats['evidence_lengths']['max']} |
| Test | {test_stats['evidence_lengths']['mean']:.1f} | {test_stats['evidence_lengths']['median']:.1f} | {test_stats['evidence_lengths']['min']} | {test_stats['evidence_lengths']['max']} |

**Observation:**
- Statement length: {'Short' if dev_stats['statement_lengths']['mean'] < 20 else 'Medium' if dev_stats['statement_lengths']['mean'] < 40 else 'Long'} (avg ~{dev_stats['statement_lengths']['mean']:.0f} words)
- Evidence length: {'Short' if dev_stats['evidence_lengths']['mean'] < 50 else 'Medium' if dev_stats['evidence_lengths']['mean'] < 150 else 'Long'} (avg ~{dev_stats['evidence_lengths']['mean']:.0f} words)
- Evidence có sẵn trong dataset → **No Brave Search needed for baseline evaluation**

---

## 💡 Key Insights for Evaluation

### Adapter 4→3 Strategy
- Model: PhoBERT-NLI (4 labels: E/N/C/OTHER)
- Benchmark: ViFactCheck (3 labels: Support/Refute/NEI)
- Mapping:
  - `Support = Entailment`
  - `Refute = Contradiction`
  - `NEI = logsumexp(Neutral, OTHER)`

### Evaluation Options
- **Option A (Baseline):** Use existing evidence → No retrieval cost
- **Option B (Full E2E):** Test với Brave Search → Full pipeline test

### Expected Challenges
- NEI class may be harder if model relies heavily on OTHER signal
- Evidence quality varies → May affect NLI confidence
- Vietnamese-specific language patterns

---

## 🎯 Next Steps

1. **Phase 3.2:** Temperature calibration on dev set
2. **Phase 3.2:** Evaluate PhoBERT-NLI + Adapter on test set
3. **Metrics:** Accuracy, Macro-F1, Per-class F1, ECE, Confusion Matrix
4. **Optional:** Compare Option A (evidence sẵn) vs Option B (Brave Search)

---

## 📁 Files

- **Dataset loader:** `src/pipeline/fact_checking/vifactcheck_dataset.py`
- **Analysis script:** `src/pipeline/fact_checking/analyze_vifactcheck.py`
- **Evaluation script:** `src/evaluation/eval_vifactcheck_adapter.py` (Phase 3.2)

---

**Status:** ✅ Phase 3.1 Complete - Ready for Phase 3.2 Evaluation
"""
    
    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"Report saved to: {output_path}")


def main():
    """Main analysis pipeline."""
    logger.info("Starting ViFactCheck dataset analysis...")
    
    # Load datasets
    logger.info("\n[1/3] Loading dev set...")
    dev_data = load_vifactcheck("dev")
    dev_stats = compute_statistics(dev_data)
    
    logger.info("\n[2/3] Loading test set...")
    test_data = load_vifactcheck("test")
    test_stats = compute_statistics(test_data)
    
    # Generate report
    logger.info("\n[3/3] Generating report...")
    output_path = os.path.join(project_root, "docs", "VIFACTCHECK_ANALYSIS.md")
    
    # Need pandas for timestamp
    try:
        import pandas as pd
        globals()['pd'] = pd
    except ImportError:
        logger.warning("pandas not found, using simple timestamp")
        from datetime import datetime
        class SimplePD:
            class Timestamp:
                @staticmethod
                def now():
                    class DT:
                        @staticmethod
                        def strftime(fmt):
                            return datetime.now().strftime(fmt)
                    return DT
        globals()['pd'] = SimplePD
    
    generate_report(dev_stats, test_stats, output_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 VIFACTCHECK DATASET ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\n✅ Dev set: {dev_stats['total_samples']} samples")
    print(f"   - Support: {dev_stats['label_distribution']['Support']}")
    print(f"   - Refute: {dev_stats['label_distribution']['Refute']}")
    print(f"   - NEI: {dev_stats['label_distribution']['NEI']}")
    
    print(f"\n✅ Test set: {test_stats['total_samples']} samples")
    print(f"   - Support: {test_stats['label_distribution']['Support']}")
    print(f"   - Refute: {test_stats['label_distribution']['Refute']}")
    print(f"   - NEI: {test_stats['label_distribution']['NEI']}")
    
    print(f"\n📄 Report saved: {output_path}")
    print("\n✅ Phase 3.1 Complete! Ready for Phase 3.2 Evaluation")


if __name__ == "__main__":
    main()
