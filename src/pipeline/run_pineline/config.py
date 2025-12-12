"""
Config cho ViFactCheck Pipeline - Clean Version

Author: Lockdown
Date: Nov 27, 2025
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Root project path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


@dataclass
class PipelineConfig:
    """Configuration cho ViFactCheck Pipeline."""
    
    # ===== Model Checkpoints =====
    # Updated: Dec 01, 2025 - PyVi preprocessed model (83.48% accuracy)
    factcheck_3label_checkpoint: str = str(
        PROJECT_ROOT / "results" / "fact_checking" / "pyvi" / "checkpoints" / "best_model_pyvi.pt"
    )
    phobert_model_name: str = "vinai/phobert-base"
    
    # ===== Debate System =====
    use_debate: bool = True
    use_async_debate: bool = True
    hybrid_enabled: bool = True  # DOWN Framework: Skip debate if model confidence >= threshold
    # Note: Hunter auto-runs when debate is enabled and context is provided
    
    # ===== Output & Logging =====
    output_dir: str = str(PROJECT_ROOT / "results" / "vifactcheck")
    log_level: str = "INFO"
    save_intermediate: bool = False
    
    # ===== Device =====
    device: str = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    def __post_init__(self):
        """Tạo output dir nếu chưa có."""
        os.makedirs(self.output_dir, exist_ok=True)


# Singleton config instance
_default_config = None

def get_config() -> PipelineConfig:
    """Get default pipeline config"""
    global _default_config
    if _default_config is None:
        _default_config = PipelineConfig()
    return _default_config

def set_config(config: PipelineConfig):
    """Set custom pipeline config"""
    global _default_config
    _default_config = config
