"""
Debate System - Multi-agent debate for fact-checking
Implements adaptive rounds với early stopping strategies.

Author: Lockdown
Date: Nov 10, 2025
"""

from .debator import Debator, DebateArgument
from .judge import Judge, FinalVerdict, JudgeR2Anchor
from .orchestrator import AdaptiveDebateOrchestrator
from .llm_client import LLMClient
from .hybrid_strategy import HybridStrategy, HybridResult

__all__ = [
    'Debator',
    'DebateArgument',
    'Judge',
    'FinalVerdict',
    'JudgeR2Anchor',
    'AdaptiveDebateOrchestrator',
    'LLMClient',
    'HybridStrategy',
    'HybridResult'
]
