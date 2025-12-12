"""
Calibration metrics for 3-label classification.
Computes ECE, MCE, and Brier score from pipeline results.
"""

import numpy as np
from typing import Dict, List, Any


def compute_calibration_metrics(results: List[Dict[str, Any]], 
                                T: float = 1.0, 
                                n_bins: int = 10) -> Dict:
    """Tính calibration metrics (ECE, MCE, Brier) từ pipeline results.
    
    Args:
        results: List of sample results from pipeline
        T: Temperature for calibration (not used in basic ECE, kept for compatibility)
        n_bins: Số bins cho ECE/MCE
    
    Returns:
        Dict chứa ece, mce, brier_score cho cả model và final
    """
    if not results:
        return {
            'model': {'ece': 0.0, 'mce': 0.0, 'brier_score': 0.0},
            'final': {'ece': 0.0, 'mce': 0.0, 'brier_score': 0.0}
        }
    
    # Extract data from results
    model_correct = []
    model_confidence = []
    final_correct = []
    final_confidence = []
    
    for r in results:
        # Model predictions
        model_correct.append(1 if r.get('model_correct', False) else 0)
        # Get max probability as confidence
        probs = r.get('verdict_3label_probs', {})
        if probs:
            model_confidence.append(max(probs.values()))
        else:
            model_confidence.append(0.5)
        
        # Final predictions (after debate)
        final_correct.append(1 if r.get('final_correct', False) else 0)
        debate_info = r.get('debate_info', {}) or r.get('debate_result', {})
        if debate_info:
            final_confidence.append(debate_info.get('confidence', 0.5))
        else:
            final_confidence.append(model_confidence[-1])
    
    model_correct = np.array(model_correct)
    model_confidence = np.array(model_confidence)
    final_correct = np.array(final_correct)
    final_confidence = np.array(final_confidence)
    
    return {
        'model': _compute_ece_mce_brier(model_correct, model_confidence, n_bins),
        'final': _compute_ece_mce_brier(final_correct, final_confidence, n_bins)
    }


def _compute_ece_mce_brier(y_true: np.ndarray, y_prob: np.ndarray, 
                           n_bins: int = 10) -> Dict:
    """Helper function to compute ECE, MCE, Brier score."""
    # Brier score
    brier = float(np.mean((y_prob - y_true) ** 2))
    
    # ECE & MCE
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    mce = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Samples trong bin này
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            
            # Calibration error
            cal_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
            ece += prop_in_bin * cal_error
            mce = max(mce, cal_error)
    
    return {
        'ece': float(ece),
        'mce': float(mce),
        'brier_score': brier
    }
