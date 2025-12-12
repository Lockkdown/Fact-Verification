"""
XAI Module for PhoBERT (Fast Path)
Generates structured explanations with 4 fields:
- Claim highlight
- Evidence highlight  
- Relationship
- Natural explanation

Based on:
- Atanasova et al. (2020): Extractive explanations
- JustiLM (2023): Atomic justification
- Word-level Diff for conflict detection
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from transformers import AutoTokenizer, AutoModel
from transformers.utils import logging as hf_logging
import re
from difflib import SequenceMatcher

hf_logging.set_verbosity_error()

# Vietnamese word segmentation
try:
    from pyvi import ViTokenizer
    PYVI_AVAILABLE = True
except ImportError:
    PYVI_AVAILABLE = False
    print("Warning: PyVi not available. Word segmentation may be inaccurate.")


class PhoBERTXAI:
    """
    XAI generator for PhoBERT using:
    - Semantic similarity for evidence highlight
    - Attention weights for claim highlight
    - Template-based natural explanation
    """
    
    def __init__(self, model, tokenizer, max_length: int = 256):
        """
        Args:
            model: PhoBERTClassifier instance
            tokenizer: AutoTokenizer instance
            max_length: Maximum sequence length
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {0: "SUPPORTS", 1: "REFUTES", 2: "NEI"}
        
        # Set model to eval mode
        self.model.eval()
        
    def _get_embeddings(
        self, 
        text: str, 
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get PhoBERT embeddings for a text.
        
        Args:
            text: Input text
            return_attention: Whether to return attention weights
            
        Returns:
            embeddings: [CLS] token embedding
            attention: Attention weights (if requested)
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        
        # Forward pass through PhoBERT encoder
        with torch.no_grad():
            outputs = self.model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=return_attention
            )
        
        # Get [CLS] embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: [1, hidden_size]
        
        attentions = outputs.attentions if return_attention else None
        
        return cls_embedding, attentions
    
    def _split_evidence_sentences(self, evidence: str) -> List[str]:
        """
        Split evidence into sentences.
        Simple heuristic: split by '. ' or '.\n'
        
        Args:
            evidence: Evidence text
            
        Returns:
            List of sentences
        """
        # Split by period followed by space or newline
        sentences = re.split(r'\.\s+|\.\n', evidence)
        # Filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _find_evidence_highlight(
        self, 
        claim: str, 
        evidence: str
    ) -> Tuple[str, float]:
        """
        Find the most relevant sentence in evidence using semantic similarity.
        
        Args:
            claim: Claim text
            evidence: Evidence text
            
        Returns:
            best_sentence: Most similar sentence
            similarity_score: Cosine similarity score
        """
        # Get claim embedding
        claim_emb, _ = self._get_embeddings(claim, return_attention=False)
        
        # Split evidence into sentences
        sentences = self._split_evidence_sentences(evidence)
        
        if not sentences:
            return evidence[:100], 0.0  # Fallback to first 100 chars
        
        # Compute similarity for each sentence
        best_sentence = sentences[0]
        best_score = 0.0
        
        for sentence in sentences:
            sent_emb, _ = self._get_embeddings(sentence, return_attention=False)
            
            # Cosine similarity
            similarity = F.cosine_similarity(claim_emb, sent_emb, dim=1).item()
            
            if similarity > best_score:
                best_score = similarity
                best_sentence = sentence
        
        return best_sentence, best_score
    
    def _find_claim_highlight(
        self, 
        claim: str, 
        evidence: str
    ) -> str:
        """
        Find key phrase in claim.
        Strategy: Use PyVi to segment properly, then extract core content.
        
        Args:
            claim: Claim text
            evidence: Evidence text
            
        Returns:
            Key phrase from claim
        """
        # Use PyVi for proper Vietnamese word segmentation
        if PYVI_AVAILABLE:
            # PyVi joins compound words: "Việt Nam" -> "Việt_Nam"
            segmented = ViTokenizer.tokenize(claim)
            tokens = segmented.split()
        else:
            tokens = claim.split()
        
        # Remove punctuation tokens
        tokens = [t for t in tokens if t not in ['.', ',', '!', '?', ';', ':']]
        
        # If claim is very short, return original
        if len(tokens) <= 6:
            return claim.rstrip('.,!?;:')
        
        # Decide whether to skip first token
        # Skip only if it's a common subject pronoun or generic word
        skip_first_tokens = {
            'việt_nam', 'chính_phủ', 'bộ', 'ông', 'bà', 'anh', 'chị',
            'tôi', 'chúng_tôi', 'họ', 'nó', 'đây', 'đó', 'này'
        }
        
        first_token_lower = tokens[0].lower() if tokens else ""
        
        # Start from 0 if first token is important (proper noun, number, etc.)
        if first_token_lower not in skip_first_tokens:
            start_idx = 0
        else:
            start_idx = 1
        
        # Take up to 8 tokens
        end_idx = min(start_idx + 8, len(tokens))
        
        # Ensure we have at least 4 tokens
        if end_idx - start_idx < 4:
            start_idx = max(0, end_idx - 4)
        
        highlight_tokens = tokens[start_idx:end_idx]
        
        # Convert back: replace underscore with space for display
        highlight = ' '.join(highlight_tokens)
        highlight = highlight.replace('_', ' ')
        
        # Remove trailing punctuation
        highlight = highlight.rstrip('.,!?;:')
        
        return highlight
    
    def _find_conflicting_words(
        self,
        claim: str,
        evidence: str
    ) -> Dict[str, List[str]]:
        """
        Find words that are different/conflicting between claim and evidence.
        Uses word-level diff to identify contradictions.
        
        Args:
            claim: Claim text
            evidence: Evidence text
            
        Returns:
            Dictionary with:
            - claim_diff: Words in claim not in evidence (potential conflicts)
            - evidence_diff: Words in evidence not in claim (potential conflicts)
        """
        # Normalize and segment text using PyVi
        def normalize_and_segment(text: str) -> List[str]:
            # Use PyVi for proper Vietnamese word segmentation
            if PYVI_AVAILABLE:
                # PyVi joins compound words with underscore: "Việt Nam" -> "Việt_Nam"
                segmented = ViTokenizer.tokenize(text.lower())
                # Split by space to get tokens (compound words have underscores)
                words = segmented.split()
            else:
                # Fallback: simple split
                text = re.sub(r'[^\w\s]', ' ', text.lower())
                words = text.split()
            
            # Remove punctuation from each word
            words = [re.sub(r'[^\w_]', '', w) for w in words]
            
            # Filter stopwords (Vietnamese common words)
            stopwords = {'là', 'và', 'của', 'có', 'được', 'trong', 'để', 'với', 
                        'cho', 'này', 'từ', 'các', 'một', 'những', 'đã', 'đang',
                        'sẽ', 'không', 'cũng', 'như', 'theo', 'về', 'trên', 'khi',
                        'vào', 'ra', 'lên', 'xuống', 'mà', 'nếu', 'thì', 'hay',
                        'hoặc', 'nhưng', 'vì', 'bởi', 'do', 'nên', 'tại', 'đến',
                        'ngày', 'tháng', 'năm', 'lần', 'thứ'}
            return [w for w in words if w not in stopwords and len(w) > 1]
        
        claim_words = set(normalize_and_segment(claim))
        evidence_words = set(normalize_and_segment(evidence))
        
        # Find words unique to each
        claim_only = claim_words - evidence_words
        evidence_only = evidence_words - claim_words
        
        # Filter to keep meaningful differences (numbers, key nouns)
        def is_meaningful(word: str) -> bool:
            # Keep numbers
            if any(c.isdigit() for c in word):
                return True
            # Keep longer words (likely meaningful nouns)
            if len(word) >= 3:
                return True
            return False
        
        claim_diff = [w for w in claim_only if is_meaningful(w)]
        evidence_diff = [w for w in evidence_only if is_meaningful(w)]
        
        return {
            "claim_diff": claim_diff,
            "evidence_diff": evidence_diff
        }
    
    def _find_key_conflict(
        self,
        claim: str,
        evidence: str
    ) -> Tuple[str, str]:
        """
        Find the key conflicting phrases between claim and evidence.
        Uses pattern matching for common Vietnamese conflict patterns.
        
        Returns:
            Tuple of (claim_conflict, evidence_conflict)
        """
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()
        
        # Define common conflict patterns in Vietnamese
        # Format: (claim_patterns, evidence_patterns, description)
        conflict_patterns = [
            # Ordinal number conflicts
            (['thứ hai', 'thứ 2'], ['đầu tiên', 'thứ nhất', 'thứ 1'], 'ordinal'),
            (['thứ ba', 'thứ 3'], ['đầu tiên', 'thứ hai', 'thứ 2'], 'ordinal'),
            (['đầu tiên', 'thứ nhất'], ['thứ hai', 'thứ 2', 'thứ ba'], 'ordinal'),
            
            # Tense/status conflicts
            (['đã ban hành', 'đã thông qua', 'đã phê duyệt'], ['đang xem xét', 'dự thảo', 'chưa ban hành'], 'status'),
            (['đã', 'hoàn thành'], ['đang', 'chưa', 'sẽ'], 'tense'),
            (['sẽ'], ['đã', 'không'], 'tense'),
            
            # Quantity conflicts
            (['tăng'], ['giảm', 'không đổi'], 'direction'),
            (['giảm'], ['tăng', 'không đổi'], 'direction'),
            
            # Affirmation/negation
            (['có'], ['không có', 'chưa có'], 'existence'),
            (['không'], ['có', 'đã'], 'negation'),
        ]
        
        # Check each pattern
        for claim_patterns, evidence_patterns, _ in conflict_patterns:
            for cp in claim_patterns:
                if cp in claim_lower:
                    for ep in evidence_patterns:
                        if ep in evidence_lower:
                            return cp, ep
        
        # Fallback: Use word-level diff
        diffs = self._find_conflicting_words(claim, evidence)
        claim_conflicts = diffs["claim_diff"]
        evidence_conflicts = diffs["evidence_diff"]
        
        # Try to find meaningful pairs
        claim_conflict = ""
        evidence_conflict = ""
        
        # Priority 1: Find year conflicts (4-digit numbers starting with 19 or 20)
        claim_years = [w for w in claim_conflicts if re.match(r'^20\d{2}$|^19\d{2}$', w)]
        evidence_years = [w for w in evidence_conflicts if re.match(r'^20\d{2}$|^19\d{2}$', w)]
        
        if claim_years and evidence_years:
            return claim_years[0], evidence_years[0]
        
        # Priority 2: Other numbers (but only if both have numbers)
        claim_nums = [w for w in claim_conflicts if any(c.isdigit() for c in w) and len(w) >= 2]
        evidence_nums = [w for w in evidence_conflicts if any(c.isdigit() for c in w) and len(w) >= 2]
        
        if claim_nums and evidence_nums:
            return claim_nums[0], evidence_nums[0]
        
        # Priority 3: Longest meaningful words
        if claim_conflicts:
            claim_conflict = max(claim_conflicts, key=len)
        if evidence_conflicts:
            evidence_conflict = max(evidence_conflicts, key=len)
        
        return claim_conflict, evidence_conflict
    
    def _generate_natural_explanation(
        self,
        claim_highlight: str,
        evidence_highlight: str,
        relationship: str
    ) -> str:
        """
        Generate natural language explanation using template.
        
        Args:
            claim_highlight: Key phrase from claim
            evidence_highlight: Key sentence from evidence
            relationship: SUPPORTS / REFUTES / NEI
            
        Returns:
            Natural explanation
        """
        templates = {
            "SUPPORTS": (
                "Bằng chứng cung cấp thông tin phù hợp với tuyên bố." +
                (f" (Trích: '{evidence_highlight}')" if evidence_highlight else "")
            ),
            "REFUTES": (
                "Bằng chứng cho thấy nội dung trong tuyên bố không đúng với thông tin được nêu." +
                (f" (Trích: '{evidence_highlight}')" if evidence_highlight else "")
            ),
            "NEI": "Hiện tại, bằng chứng được cung cấp chưa đủ để kết luận tuyên bố đúng hay sai."
        }
        
        return templates.get(relationship, "Không đủ thông tin để giải thích.")
    
    def generate_xai(
        self,
        claim: str,
        evidence: str,
        model_verdict: Optional[str] = None,
        model_probs: Optional[Dict[str, float]] = None
    ) -> Dict[str, str]:
        """
        Generate XAI with 4 fields for Fast Path.
        
        Args:
            claim: Claim text
            evidence: Evidence text
            model_verdict: Model's predicted verdict (optional, will compute if not provided)
            model_probs: Model's prediction probabilities (optional)
            
        Returns:
            Dictionary with 4 XAI fields:
            - claim_highlight
            - evidence_highlight
            - relationship
            - natural_explanation
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
                logits = self.model(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"]
                )
                probs = F.softmax(logits, dim=1)[0]
                pred_label = torch.argmax(probs).item()
                model_verdict = self.label_map[pred_label]
        
        # 1. Find evidence highlight (semantic similarity)
        evidence_highlight, similarity = self._find_evidence_highlight(claim, evidence)
        
        # 2. Claim highlight removed from schema (Dec 2025)
        claim_highlight = ""
        
        # 3. Relationship = model verdict
        relationship = model_verdict
        
        # 4. Find conflicting words (ONLY for REFUTES cases)
        claim_conflict = ""
        evidence_conflict = ""
        
        if relationship == "REFUTES":
            claim_conflict, evidence_conflict = self._find_key_conflict(claim, evidence)
        
        # 5. Generate natural explanation (use conflict words if available for REFUTES)
        if relationship == "REFUTES" and claim_conflict and evidence_conflict:
            claim_display = claim_conflict.replace('_', ' ')
            evidence_display = evidence_conflict.replace('_', ' ')
            natural_explanation = (
                f"Tuyên bố nêu '{claim_display}' nhưng bằng chứng cho thấy '{evidence_display}'. "
                f"Hai thông tin này mâu thuẫn trực tiếp."
            )
        else:
            natural_explanation = self._generate_natural_explanation(
                claim_highlight,
                evidence_highlight,
                relationship
            )
        
        return {
            "relationship": relationship,
            "natural_explanation": natural_explanation,
            "similarity_score": similarity,
            # Conflict words only for REFUTES
            "claim_conflict_word": claim_conflict,
            "evidence_conflict_word": evidence_conflict,
        }


def load_xai_model(
    model_path: str,
    phobert_name: str = "vinai/phobert-base",
    device: str = "cpu"
) -> PhoBERTXAI:
    """
    Load PhoBERT model and create XAI instance.
    
    Args:
        model_path: Path to trained model checkpoint
        phobert_name: PhoBERT model name
        device: Device to load model on
        
    Returns:
        PhoBERTXAI instance
    """
    from .model import PhoBERTFactCheck
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(phobert_name)
    
    # Load model
    model = PhoBERTFactCheck(
        pretrained_name=phobert_name,
        num_classes=3,
        dropout_rate=0.35
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create XAI instance
    xai = PhoBERTXAI(model, tokenizer)
    
    return xai
