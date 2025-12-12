"""
Phase 3 – Inference Pipeline (Article → Verdict)

Main orchestrator cho pipeline tổng thể từ article text → claim verdicts → article verdict.
"""

from .article_pipeline import ViFactCheckPipeline
from .config import PipelineConfig

__all__ = ["ViFactCheckPipeline", "PipelineConfig"]
