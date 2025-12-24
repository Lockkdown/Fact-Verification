"""
ViFactCheck Pipeline - Clean Version

Pipeline đơn giản cho ViFactCheck dataset:
- Input: (Statement, Evidence) từ dataset
- Output: Verdict (Support/Refute/NEI) + Debate reasoning

Author: Lockdown
Date: Nov 27, 2025
"""

import torch
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
import unicodedata

# Fix for "Event loop is closed" error when using asyncio.run() multiple times
import nest_asyncio
nest_asyncio.apply()

# PyVi for word segmentation (MUST match training preprocessing)
try:
    from pyvi import ViTokenizer
    PYVI_AVAILABLE = True
except ImportError:
    PYVI_AVAILABLE = False
    ViTokenizer = None

from src.pipeline.debate.orchestrator import AdaptiveDebateOrchestrator
from src.pipeline.debate.debator import Evidence as DebateEvidence

# PhoBERT XAI for fast path (Dec 2025)
try:
    from src.pipeline.fact_checking.xai_phobert import PhoBERTXAI
    PHOBERT_XAI_AVAILABLE = True
except ImportError:
    PHOBERT_XAI_AVAILABLE = False
    PhoBERTXAI = None

from .config import PipelineConfig, get_config
from .utils import load_factcheck_model_3label, Timer

logger = logging.getLogger(__name__)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


@dataclass
class EvidenceItem:
    """Simple evidence container for ViFactCheck."""
    text: str
    source: str = "vifactcheck"


class ViFactCheckPipeline:
    """
    Pipeline cho ViFactCheck dataset evaluation.
    
    Flow: Statement + Evidence → PhoBERT Verdict → (Optional) Debate → Final Verdict
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Args:
            config: PipelineConfig, nếu None sẽ dùng default config
        """
        self.config = config or get_config()
        
        # Load PhoBERT model for 3-label classification
        logger.info("Loading ViFactCheck 3-label model...")
        self.model, self.tokenizer, self.device = load_factcheck_model_3label(self.config)
        
        # Debate system (optional)
        self.debate_orchestrator = None
        if self.config.use_debate:
            logger.info("Initializing Debate Orchestrator...")
            self.debate_orchestrator = AdaptiveDebateOrchestrator()
            
            # Override hybrid_enabled from CLI config (--full-debate vs --hybrid-debate)
            cli_hybrid_enabled = getattr(self.config, 'hybrid_enabled', True)
            if not cli_hybrid_enabled:
                self.debate_orchestrator.hybrid_enabled = False
                logger.info("⚠️ Hybrid Strategy: DISABLED (--full-debate mode)")
        
        # PhoBERT XAI for fast path (Dec 2025)
        self.phobert_xai = None
        if PHOBERT_XAI_AVAILABLE:
            try:
                self.phobert_xai = PhoBERTXAI(self.model, self.tokenizer)
                logger.info("✅ PhoBERT XAI initialized for fast path")
            except Exception as e:
                logger.warning(f"⚠️ PhoBERT XAI not available: {e}")
        
        logger.info("ViFactCheckPipeline initialized successfully.")
    
    def predict(
        self,
        statement: str,
        evidence: str,
        use_debate: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Predict verdict cho một cặp (statement, evidence) using Gold Evidence.
        
        Args:
            statement: Claim/Statement cần verify
            evidence: Gold evidence text from dataset
            use_debate: Override config.use_debate nếu cần
        
        Returns:
            Dict với model_verdict, final_verdict, confidence, debate_info, etc.
        """
        if use_debate is None:
            use_debate = self.config.use_debate
        
        result = {
            "statement": statement,
            "evidence": evidence[:200] + "..." if len(evidence) > 200 else evidence,
            "model_verdict": None,
            "model_confidence": None,
            "model_probs": None,
            "debate_verdict": None,
            "debate_confidence": None,
            "debate_reasoning": None,
            "debate_round_1_verdicts": None,
            "debate_metrics": None,  # New field for full metrics
            "final_verdict": None
        }
        
        # Step 1: PhoBERT prediction
        with Timer("model_prediction"):
            model_verdict, confidence, probs = self._predict_verdict(statement, evidence)
        
        result["model_verdict"] = model_verdict
        result["model_confidence"] = round(confidence, 4)
        result["model_probs"] = {k: round(v, 6) for k, v in probs.items()}
        
        logger.info(f"Model: {model_verdict} (conf={confidence:.2f})")
        
        # Step 2: Debate (optional)
        # ⚡ Hybrid Strategy Check (DOWN Framework)
        # Check both config.hybrid_enabled (from CLI) and orchestrator.hybrid_enabled (from JSON)
        skip_debate = False
        hybrid_enabled = getattr(self.config, 'hybrid_enabled', True) and self.debate_orchestrator.hybrid_enabled
        if use_debate and self.debate_orchestrator and hybrid_enabled:
            # Use dummy debate values for initial check
            hybrid_decision = self.debate_orchestrator.hybrid_strategy.decide(
                model_verdict=model_verdict,
                model_probs=probs,
                debate_verdict="NEI", 
                debate_confidence=0.0
            )
            
            if hybrid_decision.decision_source == "MODEL_HIGH_CONF":
                logger.info(f"🚀 Hybrid Strategy: Skipping Debate (Model Conf {confidence:.2f} >= {self.debate_orchestrator.hybrid_threshold})")
                skip_debate = True
                result["hybrid_info"] = {
                    "source": "MODEL_HIGH_CONF",
                    "threshold": self.debate_orchestrator.hybrid_threshold,
                    "reasoning": hybrid_decision.reasoning
                }

        if use_debate and self.debate_orchestrator and not skip_debate:
            # Prepare gold evidence from dataset
            debate_evidence = DebateEvidence(
                text=evidence,
                source="vifactcheck",
                rank=1,
                nli_score={"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0, "other": 0.0},
                relevance_score=1.0
            )
            
            with Timer("debate"):
                if self.config.use_async_debate:
                    logger.info(f"⚡ Using ASYNC debate (Gold Evidence)")
                    debate_result = asyncio.run(
                        self.debate_orchestrator.debate_async(
                            claim=statement,
                            evidences=[debate_evidence],
                            model_verdict=model_verdict,
                            model_confidence=confidence
                        )
                    )
                else:
                    logger.info(f"⏱️ Using SYNC debate (Gold Evidence)")
                    debate_result = self.debate_orchestrator.debate(
                        claim=statement,
                        evidences=[debate_evidence],
                        model_verdict=model_verdict,
                        model_confidence=confidence
                    )
            
            result["debate_verdict"] = debate_result.verdict
            result["debate_confidence"] = round(debate_result.confidence, 4)
            result["debate_reasoning"] = debate_result.reasoning
            result["debate_round_1_verdicts"] = debate_result.round_1_verdicts
            # Save all rounds verdicts if available (for visualization)
            result["debate_all_rounds_verdicts"] = getattr(debate_result, "all_rounds_verdicts", None)
            result["final_verdict"] = debate_result.verdict
            
            # Store full debate metrics for evaluation
            result["debate_metrics"] = {
                "rounds_used": debate_result.rounds_used,
                "early_stopped": debate_result.early_stopped,
                "stop_reason": debate_result.stop_reason,
                "mvp_agent": debate_result.mvp_agent,
                "debator_agreements": debate_result.debator_agreements,
                # New metrics (Dec 2025)
                "consensus_round": getattr(debate_result, "consensus_round", None),
                "decision_path": getattr(debate_result, "decision_path", None)
            }
            
            # Store XAI dict (Dec 2025)
            result["debate_xai"] = getattr(debate_result, "xai_dict", None)
            
            logger.info(f"Debate: {debate_result.verdict} (conf={debate_result.confidence:.2f})")
        else:
            result["final_verdict"] = model_verdict
            
            # Generate PhoBERT XAI for fast path (Dec 2025)
            if self.phobert_xai:
                try:
                    verdict_map = {"Support": "SUPPORTS", "Refute": "REFUTES", "NEI": "NEI"}
                    xai_verdict = verdict_map.get(model_verdict, "NEI")
                    
                    phobert_xai_result = self.phobert_xai.generate_xai(
                        claim=statement,
                        evidence=evidence,
                        model_verdict=xai_verdict
                    )
                    
                    result["debate_xai"] = {
                        "relationship": phobert_xai_result.get("relationship", "NEI"),
                        "natural_explanation": phobert_xai_result.get("natural_explanation", ""),
                        "conflict_claim": phobert_xai_result.get("claim_conflict_word", ""),
                        "conflict_evidence": phobert_xai_result.get("evidence_conflict_word", ""),
                        "source": "FAST_PATH",
                        "confidence": confidence
                    }
                except Exception as e:
                    logger.warning(f"PhoBERT XAI generation failed: {e}")
        
        return result
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text with PyVi word segmentation.
        MUST match training preprocessing (preprocessing_vifactcheck_pyvi.py).
        """
        # Unicode normalization
        text = unicodedata.normalize('NFC', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        # Word segmentation with pyvi
        if PYVI_AVAILABLE:
            text = ViTokenizer.tokenize(text)
        else:
            logger.warning("⚠️ PyVi not available! Inference will use raw text (MISMATCH with training!)")
        return text.strip()
    
    def _predict_verdict(self, statement: str, evidence: str) -> tuple[str, float, dict]:
        """
        Predict verdict using PhoBERT model.
        
        Returns:
            (verdict, confidence, probs_dict)
        """
        # ⚡ CRITICAL: Preprocess with PyVi to match training data
        statement_processed = self._preprocess_text(statement)
        evidence_processed = self._preprocess_text(evidence)
        
        # Tokenize
        inputs = self.tokenizer(
            statement_processed,
            evidence_processed,
            max_length=256,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
            if k in ("input_ids", "attention_mask")
        }
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            probs = torch.softmax(logits, dim=-1).squeeze()
            pred_label = torch.argmax(probs, dim=-1).item()
            confidence = probs[pred_label].item()
        
        # Map to labels
        label_map = {0: "Support", 1: "Refute", 2: "NEI"}
        probs_dict = {
            "Support": float(probs[0].item()),
            "Refute": float(probs[1].item()),
            "NEI": float(probs[2].item())
        }
        
        return label_map[pred_label], confidence, probs_dict
    
    async def predict_async(
        self,
        statement: str,
        evidence: str,
        use_debate: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Async version of predict for batch processing with Gold Evidence.
        """
        if use_debate is None:
            use_debate = self.config.use_debate
        
        result = {
            "statement": statement,
            "evidence": evidence[:200] + "..." if len(evidence) > 200 else evidence,
            "model_verdict": None,
            "model_confidence": None,
            "model_probs": None,
            "debate_verdict": None,
            "debate_confidence": None,
            "debate_reasoning": None,
            "debate_round_1_verdicts": None,
            "debate_all_rounds_verdicts": None,
            "debate_metrics": None,
            "final_verdict": None
        }
        
        # Step 1: PhoBERT prediction (sync - fast enough)
        model_verdict, confidence, probs = self._predict_verdict(statement, evidence)
        
        result["model_verdict"] = model_verdict
        result["model_confidence"] = round(confidence, 4)
        result["model_probs"] = {k: round(v, 6) for k, v in probs.items()}
        
        # Step 2: Debate (async)
        # ⚡ Hybrid Strategy Check (DOWN Framework)
        skip_debate = False
        hybrid_enabled = getattr(self.config, 'hybrid_enabled', True) and self.debate_orchestrator.hybrid_enabled
        if use_debate and self.debate_orchestrator and hybrid_enabled:
            hybrid_decision = self.debate_orchestrator.hybrid_strategy.decide(
                model_verdict=model_verdict,
                model_probs=probs,
                debate_verdict="NEI",
                debate_confidence=0.0
            )
            
            if hybrid_decision.decision_source == "MODEL_HIGH_CONF":
                logger.info(f"🚀 Hybrid: SKIP Debate (Conf {confidence:.2f} >= {self.debate_orchestrator.hybrid_threshold})")
                print(f"🚀 SKIP: {model_verdict} (conf={confidence:.2f})")  # Force print for batch mode
                skip_debate = True
                result["hybrid_info"] = {
                    "source": "MODEL_HIGH_CONF",
                    "threshold": self.debate_orchestrator.hybrid_threshold,
                    "skipped": True
                }
        
        if use_debate and self.debate_orchestrator and not skip_debate:
            # Prepare gold evidence from dataset
            debate_evidence = DebateEvidence(
                text=evidence,
                source="vifactcheck",
                rank=1,
                nli_score={"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0, "other": 0.0},
                relevance_score=1.0
            )
            
            # Call async debate directly (no asyncio.run needed)
            debate_result = await self.debate_orchestrator.debate_async(
                claim=statement,
                evidences=[debate_evidence],
                model_verdict=model_verdict,
                model_confidence=confidence
            )
            
            result["debate_verdict"] = debate_result.verdict
            result["debate_confidence"] = round(debate_result.confidence, 4)
            result["debate_reasoning"] = debate_result.reasoning
            result["debate_round_1_verdicts"] = debate_result.round_1_verdicts
            result["debate_all_rounds_verdicts"] = getattr(debate_result, "all_rounds_verdicts", None)
            result["final_verdict"] = debate_result.verdict
            
            result["debate_metrics"] = {
                "rounds_used": debate_result.rounds_used,
                "early_stopped": debate_result.early_stopped,
                "stop_reason": debate_result.stop_reason,
                "mvp_agent": debate_result.mvp_agent,
                "debator_agreements": debate_result.debator_agreements
            }
            result["hybrid_info"] = {"source": "DEBATE", "skipped": False}
            
            # Store XAI dict (Dec 2025)
            result["debate_xai"] = getattr(debate_result, "xai_dict", None)
        else:
            result["final_verdict"] = model_verdict
            
            # Generate PhoBERT XAI for fast path (Dec 2025)
            if self.phobert_xai:
                try:
                    # Map verdict to XAI format
                    verdict_map = {"Support": "SUPPORTS", "Refute": "REFUTES", "NEI": "NEI"}
                    xai_verdict = verdict_map.get(model_verdict, "NEI")
                    
                    phobert_xai_result = self.phobert_xai.generate_xai(
                        claim=statement,
                        evidence=evidence,
                        model_verdict=xai_verdict
                    )
                    
                    result["debate_xai"] = {
                        "relationship": phobert_xai_result.get("relationship", "NEI"),
                        "natural_explanation": phobert_xai_result.get("natural_explanation", ""),
                        "conflict_claim": phobert_xai_result.get("claim_conflict_word", ""),
                        "conflict_evidence": phobert_xai_result.get("evidence_conflict_word", ""),
                        "source": "FAST_PATH",
                        "generated_by": "phobert_xai"
                    }
                except Exception as e:
                    logger.warning(f"PhoBERT XAI generation failed: {e}")
        
        return result
    
    def batch_predict(
        self,
        samples: List[Dict[str, str]],
        use_debate: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Predict cho batch samples.
        
        Args:
            samples: List of {"statement": str, "evidence": str}
            use_debate: Override config
        
        Returns:
            List of prediction results
        """
        results = []
        for i, sample in enumerate(samples):
            logger.info(f"\n{'='*60}\nSample {i+1}/{len(samples)}\n{'='*60}")
            result = self.predict(
                statement=sample["statement"],
                evidence=sample["evidence"],
                use_debate=use_debate
            )
            results.append(result)
        return results


    async def shutdown(self):
        """Cleanly shutdown pipeline resources (close sessions)."""
        if self.debate_orchestrator:
            logger.info("🔌 Shutting down Debate Orchestrator sessions...")
            await self.debate_orchestrator.close()


# Backward compatibility alias
ArticlePipeline = ViFactCheckPipeline
