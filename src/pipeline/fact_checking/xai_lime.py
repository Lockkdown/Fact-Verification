"""
LIME-based XAI Module for PhoBERT
Uses LIME (Local Interpretable Model-agnostic Explanations) for better explanations.

Based on research showing LIME achieves 0.97 Human Agreement score on BERT models,
significantly better than attention-based methods (0.07-0.17).

Reference: arXiv:2501.15374 - Evaluating XAI Techniques for Encoder-Based Language Models
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from lime.lime_text import LimeTextExplainer
import re
from pathlib import Path
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

# Vietnamese word segmentation
try:
    from pyvi import ViTokenizer
    PYVI_AVAILABLE = True
except ImportError:
    PYVI_AVAILABLE = False


_STOPWORDS_CACHE: Optional[set] = None


def _load_vietnamese_stopwords() -> set:
    global _STOPWORDS_CACHE
    if _STOPWORDS_CACHE is not None:
        return _STOPWORDS_CACHE

    stopwords: set = set()

    vendor_path = Path(__file__).resolve().parents[3] / "resources" / "vietnamese-stopwords.txt"
    if vendor_path.exists():
        try:
            for line in vendor_path.read_text(encoding="utf-8").splitlines():
                w = line.strip().lower()
                if not w:
                    continue
                if " " in w or "\t" in w:
                    continue
                stopwords.add(w)
        except Exception:
            stopwords = set()

    if not stopwords:
        stopwords = {
            "là", "và", "của", "có", "được", "trong", "để", "với",
            "cho", "này", "từ", "các", "một", "những", "đã", "đang",
            "sẽ", "không", "cũng", "như", "theo", "về", "trên", "khi",
        }

    _STOPWORDS_CACHE = stopwords
    return stopwords


class PhoBERTLimeXAI:
    """
    LIME-based XAI for PhoBERT fact-checking model.
    Provides token-level importance scores for claim and evidence.
    """
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        max_length: int = 256,
        num_samples: int = 100,  # Reduced for speed
        device: str = "cpu"
    ):
        """
        Args:
            model: PhoBERTFactCheck model
            tokenizer: AutoTokenizer
            max_length: Max sequence length
            num_samples: Number of perturbations for LIME (lower = faster)
            device: cpu or cuda
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples
        self.device = device
        self.label_map = {0: "SUPPORTS", 1: "REFUTES", 2: "NEI"}
        self.label_map_reverse = {"SUPPORTS": 0, "REFUTES": 1, "NEI": 2}
        
        # LIME explainer for text
        self.explainer = LimeTextExplainer(
            class_names=["SUPPORTS", "REFUTES", "NEI"],
            split_expression=r'\s+',  # Split by whitespace
            bow=False  # Keep word order
        )
        
        self.model.eval()
    
    def _predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Prediction function for LIME.
        Takes list of text strings, returns probabilities.
        """
        # LIME calls this function many times; batching is critical for speed.
        # We keep max_length for truncation but pad only to the longest in each batch.
        batch_size = 32
        all_probs: List[np.ndarray] = []

        claims: List[str] = []
        evidences: List[str] = []
        for text in texts:
            if " [SEP] " in text:
                parts = text.split(" [SEP] ", 1)
                claims.append(parts[0])
                evidences.append(parts[1] if len(parts) > 1 else "")
            else:
                claims.append(text)
                evidences.append("")

        for i in range(0, len(texts), batch_size):
            batch_claims = claims[i:i + batch_size]
            batch_evidences = evidences[i:i + batch_size]

            encoding = self.tokenizer(
                batch_claims,
                batch_evidences,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            with torch.inference_mode():
                outputs = self.model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=-1).cpu().numpy()

            for row in probs:
                all_probs.append(row)

        return np.array(all_probs)
    
    def _get_word_importance(
        self, 
        text: str, 
        label_idx: int
    ) -> List[Tuple[str, float]]:
        """
        Get word importance scores using LIME.
        
        Args:
            text: Combined "claim [SEP] evidence" text
            label_idx: Target label index
            
        Returns:
            List of (word, importance_score) tuples
        """
        try:
            exp = self.explainer.explain_instance(
                text,
                self._predict_proba,
                num_features=20,  # Top 20 important words
                num_samples=self.num_samples,
                labels=[label_idx]
            )
            
            # Get explanation for target label
            word_scores = exp.as_list(label=label_idx)
            return word_scores
            
        except Exception as e:
            print(f"LIME error: {e}")
            return []
    
    def _extract_highlights(
        self,
        claim: str,
        evidence: str,
        word_scores: List[Tuple[str, float]],
        top_k: int = 5
    ) -> Tuple[str, str, str, str]:
        """
        Extract highlight phrases from word importance scores.
        
        Returns:
            claim_highlight, evidence_highlight, claim_conflict, evidence_conflict
        """
        if not word_scores:
            return "", "", "", ""
        
        # Filter out special tokens and short words
        skip_words = {"[SEP]", "[CLS]", "[PAD]", ".", ",", ":", ";", "!", "?", "-"}
        filtered_scores = [
            (w, s) for w, s in word_scores 
            if w not in skip_words and len(w.replace("_", "")) > 1
        ]
        
        # Separate positive and negative importance words
        positive_words = [(w, s) for w, s in filtered_scores if s > 0]
        negative_words = [(w, s) for w, s in filtered_scores if s < 0]
        
        # Sort by absolute importance
        positive_words.sort(key=lambda x: -x[1])
        negative_words.sort(key=lambda x: x[1])  # Most negative first
        
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()
        
        # Find highlights - words with high positive importance
        claim_highlights = []
        evidence_highlights = []
        
        for word, score in positive_words[:top_k]:
            word_clean = word.replace("_", " ").lower().strip(".,!?")
            if word_clean in claim_lower and word_clean not in [h.lower() for h in claim_highlights]:
                claim_highlights.append(word.replace("_", " ").strip(".,!?"))
            if word_clean in evidence_lower and word_clean not in [h.lower() for h in evidence_highlights]:
                evidence_highlights.append(word.replace("_", " ").strip(".,!?"))
        
        # Find conflicts - look for antonym pairs or words only in one text
        claim_only = []  # Words in claim but not evidence
        evidence_only = []  # Words in evidence but not claim
        
        for word, score in positive_words[:top_k * 2]:
            word_clean = word.replace("_", " ").lower().strip(".,!?")
            in_claim = word_clean in claim_lower
            in_evidence = word_clean in evidence_lower
            
            if in_claim and not in_evidence:
                claim_only.append((word.replace("_", " ").strip(".,!?"), score))
            elif in_evidence and not in_claim:
                evidence_only.append((word.replace("_", " ").strip(".,!?"), score))
        
        # Build highlight strings
        claim_highlight = claim_highlights[0] if claim_highlights else ""
        evidence_highlight = evidence_highlights[0] if evidence_highlights else ""
        
        # Conflict: prioritize words unique to each text
        claim_conflict = claim_only[0][0] if claim_only else ""
        evidence_conflict = evidence_only[0][0] if evidence_only else ""
        
        # Fallback: if no unique conflicts found, use antonym detection
        if not claim_conflict or not evidence_conflict:
            antonym_pairs = [
                ("thua", "thắng"), ("thắng", "thua"),
                ("không", "có"), ("có", "không"),
                ("đúng", "sai"), ("sai", "đúng"),
                ("tăng", "giảm"), ("giảm", "tăng"),
                ("lớn", "nhỏ"), ("nhỏ", "lớn"),
                ("cao", "thấp"), ("thấp", "cao"),
            ]
            
            for w1, w2 in antonym_pairs:
                if w1 in claim_lower and w2 in evidence_lower:
                    claim_conflict = w1
                    evidence_conflict = w2
                    break
        
        return claim_highlight, evidence_highlight, claim_conflict, evidence_conflict

    def _find_conflicting_words_rule(self, claim: str, evidence: str) -> Dict[str, List[str]]:
        def normalize_and_segment(text: str) -> List[str]:
            if PYVI_AVAILABLE:
                segmented = ViTokenizer.tokenize(text.lower())
                words = segmented.split()
            else:
                text = re.sub(r'[^\w\s]', ' ', text.lower())
                words = text.split()

            words = [re.sub(r'[^\w_]', '', w) for w in words]

            stopwords = _load_vietnamese_stopwords()
            # Keep single-digit numbers (e.g., "5", "7") but filter short non-numeric words
            return [w for w in words if w not in stopwords and (len(w) > 1 or w.isdigit())]

        claim_words = set(normalize_and_segment(claim))
        evidence_words = set(normalize_and_segment(evidence))

        claim_only = claim_words - evidence_words
        evidence_only = evidence_words - claim_words

        def is_meaningful(word: str) -> bool:
            if any(c.isdigit() for c in word):
                return True
            if len(word) >= 3:
                return True
            return False

        claim_diff = [w for w in claim_only if is_meaningful(w)]
        evidence_diff = [w for w in evidence_only if is_meaningful(w)]

        return {"claim_diff": claim_diff, "evidence_diff": evidence_diff}

    def _find_key_conflict_rule(self, claim: str, evidence: str) -> Tuple[str, str]:
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()

        conflict_patterns = [
            (['thứ hai', 'thứ 2'], ['đầu tiên', 'thứ nhất', 'thứ 1'], 'ordinal'),
            (['thứ ba', 'thứ 3'], ['đầu tiên', 'thứ hai', 'thứ 2'], 'ordinal'),
            (['đầu tiên', 'thứ nhất'], ['thứ hai', 'thứ 2', 'thứ ba'], 'ordinal'),
            (['đã ban hành', 'đã thông qua', 'đã phê duyệt'], ['đang xem xét', 'dự thảo', 'chưa ban hành'], 'status'),
            (['đã', 'hoàn thành'], ['đang', 'chưa', 'sẽ'], 'tense'),
            (['sẽ'], ['đã', 'không'], 'tense'),
            (['tăng'], ['giảm', 'không đổi'], 'direction'),
            (['giảm'], ['tăng', 'không đổi'], 'direction'),
            (['có'], ['không có', 'chưa có'], 'existence'),
            (['không'], ['có', 'đã'], 'negation'),
        ]

        for claim_patterns, evidence_patterns, _ in conflict_patterns:
            for cp in claim_patterns:
                if cp in claim_lower:
                    for ep in evidence_patterns:
                        if ep in evidence_lower:
                            return cp, ep

        diffs = self._find_conflicting_words_rule(claim, evidence)
        claim_conflicts = diffs["claim_diff"]
        evidence_conflicts = diffs["evidence_diff"]

        claim_years = [w for w in claim_conflicts if re.match(r'^20\d{2}$|^19\d{2}$', w)]
        evidence_years = [w for w in evidence_conflicts if re.match(r'^20\d{2}$|^19\d{2}$', w)]
        if claim_years and evidence_years:
            return claim_years[0], evidence_years[0]

        # Match numbers (including single digits like "5", "7")
        claim_nums = [w for w in claim_conflicts if any(c.isdigit() for c in w)]
        evidence_nums = [w for w in evidence_conflicts if any(c.isdigit() for c in w)]
        if claim_nums and evidence_nums:
            return claim_nums[0], evidence_nums[0]

        claim_conflict = max(claim_conflicts, key=len) if claim_conflicts else ""
        evidence_conflict = max(evidence_conflicts, key=len) if evidence_conflicts else ""

        return claim_conflict, evidence_conflict
    
    def _generate_explanation(
        self,
        relationship: str,
        claim_highlight: str,
        evidence_highlight: str,
        claim_conflict: str,
        evidence_conflict: str,
        claim: str,
        evidence: str
    ) -> str:
        """Generate natural language explanation based on LIME results."""
        
        if relationship == "SUPPORTS":
            if claim_highlight and evidence_highlight:
                return f"Bằng chứng cung cấp thông tin phù hợp với phần '{claim_highlight}' trong tuyên bố."
            elif evidence_highlight:
                return f"Bằng chứng cho thấy tuyên bố là phù hợp (chi tiết: '{evidence_highlight}')."
            else:
                return "Bằng chứng cung cấp thông tin phù hợp với tuyên bố."
        
        elif relationship == "REFUTES":
            if claim_conflict and evidence_conflict:
                return f"Tuyên bố nêu '{claim_conflict}' nhưng bằng chứng cho thấy '{evidence_conflict}'. Hai thông tin này mâu thuẫn."
            elif claim_conflict:
                return f"Chi tiết '{claim_conflict}' trong tuyên bố không khớp với bằng chứng."
            elif evidence_conflict:
                return f"Bằng chứng về '{evidence_conflict}' cho thấy tuyên bố không chính xác."
            else:
                return "Bằng chứng cho thấy nội dung trong tuyên bố không đúng với thông tin được nêu."
        
        else:  # NEI
            if claim_highlight:
                return f"Bằng chứng hiện có chưa đề cập rõ phần '{claim_highlight}', nên chưa thể kết luận tuyên bố đúng hay sai."
            else:
                return "Hiện tại, bằng chứng được cung cấp chưa đủ để kết luận tuyên bố đúng hay sai."
    
    def generate_xai(
        self,
        claim: str,
        evidence: str,
        model_verdict: Optional[int] = None
    ) -> Dict:
        """
        Generate XAI explanation using LIME.
        
        Args:
            claim: Claim text
            evidence: Evidence text
            model_verdict: Optional pre-computed verdict (0=SUPPORTS, 1=REFUTES, 2=NEI)
            
        Returns:
            Dict with:
            - claim_highlight
            - evidence_highlight
            - relationship
            - natural_explanation
            - word_scores (LIME importance scores)
            - claim_conflict_word
            - evidence_conflict_word
        """
        # Get model prediction if not provided
        if model_verdict is None:
            encoding = self.tokenizer(
                claim,
                evidence,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self.model(
                    encoding["input_ids"].to(self.device),
                    encoding["attention_mask"].to(self.device)
                )
                probs = torch.softmax(outputs, dim=-1)
                model_verdict = torch.argmax(probs, dim=-1).item()
                confidence = probs[0, model_verdict].item()
        else:
            confidence = 1.0
        
        relationship = self.label_map[model_verdict]

        claim_highlight = ""
        evidence_highlight = ""
        word_scores: List[Tuple[str, float]] = []
        claim_conflict = ""
        evidence_conflict = ""
        
        # Hybrid: for REFUTES, override conflict words using rule-based conflict detection
        if relationship == "REFUTES":
            rule_claim_conflict, rule_evidence_conflict = self._find_key_conflict_rule(claim, evidence)
            if rule_claim_conflict and rule_evidence_conflict:
                claim_conflict = rule_claim_conflict.replace('_', ' ')
                evidence_conflict = rule_evidence_conflict.replace('_', ' ')
        
        # Generate explanation
        natural_explanation = self._generate_explanation(
            relationship,
            claim_highlight,
            evidence_highlight,
            claim_conflict,
            evidence_conflict,
            claim,
            evidence
        )
        
        return {
            "relationship": relationship,
            "natural_explanation": natural_explanation,
            "confidence": confidence,
            "claim_conflict_word": claim_conflict,
            "evidence_conflict_word": evidence_conflict,
            "word_scores": []
        }


def load_lime_xai_model(
    model_path: str,
    pretrained_name: str = "vinai/phobert-base",
    device: str = "cpu",
    num_samples: int = 100
) -> PhoBERTLimeXAI:
    """
    Load PhoBERT model and create LIME XAI instance.
    
    Args:
        model_path: Path to trained model checkpoint
        pretrained_name: PhoBERT model name
        device: Device to load model on
        num_samples: Number of LIME samples (lower = faster)
        
    Returns:
        PhoBERTLimeXAI instance
    """
    from .model import PhoBERTFactCheck
    from transformers import AutoTokenizer
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    
    # Load model
    model = PhoBERTFactCheck(
        num_classes=3,
        pretrained_name=pretrained_name
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return PhoBERTLimeXAI(
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_samples=num_samples
    )


if __name__ == "__main__":
    # Quick test
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    model_path = project_root / "results/fact_checking/pyvi/checkpoints/best_model_pyvi.pt"
    
    print("Loading LIME XAI model...")
    xai = load_lime_xai_model(str(model_path), device="cpu", num_samples=50)
    
    # Test case
    claim = "Việt Nam thua Myanmar ở lượt cuối."
    evidence = "Việt Nam thắng Myanmar 2-0 ở lượt cuối, qua đó đi tiếp với tư cách đội nhất bảng B."
    
    print(f"\nClaim: {claim}")
    print(f"Evidence: {evidence}")
    
    result = xai.generate_xai(claim, evidence)
    
    print(f"\n--- LIME XAI Result ---")
    print(f"Relationship: {result['relationship']}")
    print(f"Claim Conflict: {result['claim_conflict_word']}")
    print(f"Evidence Conflict: {result['evidence_conflict_word']}")
    print(f"Explanation: {result['natural_explanation']}")
    print(f"\nTop Word Scores:")
    for word, score in result['word_scores']:
        print(f"  {word}: {score:.4f}")
