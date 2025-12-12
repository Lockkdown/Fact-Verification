"""
Test script cho Adaptive Debate System.

Usage:
    python test_debate.py

Author: Lockdown
Date: Nov 10, 2025
"""

import sys
import logging
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.debate import AdaptiveDebateOrchestrator
from src.pipeline.debate.debator import Evidence

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_single_claim(models_config_path: str = None, debate_config_path: str = None, environment: str = None):
    """Test với 1 claim đơn giản."""
    
    logger.info("="*80)
    logger.info("TEST: Single Claim Debate")
    logger.info("="*80)
    
    # Initialize orchestrator
    orchestrator = AdaptiveDebateOrchestrator(
        models_config_path=models_config_path,
        debate_config_path=debate_config_path,
        environment=environment
    )
    
    # Test claim
    claim = "GDP Việt Nam năm 2024 đạt mức tăng trưởng 7.09%."
    
    # Mock evidences (thực tế sẽ từ Brave Search + Reranker)
    evidences = [
        Evidence(
            text="Tổng cục Thống kê công bố GDP Việt Nam năm 2024 tăng 7.09% so với năm 2023.",
            source="https://gso.gov.vn",
            rank=1,
            nli_score={
                "entailment": 0.92,
                "neutral": 0.05,
                "contradiction": 0.02,
                "other": 0.01
            }
        ),
        Evidence(
            text="Theo báo cáo của Ngân hàng Thế giới, tăng trưởng kinh tế Việt Nam năm 2024 đạt 7.1%.",
            source="https://worldbank.org",
            rank=2,
            nli_score={
                "entailment": 0.88,
                "neutral": 0.08,
                "contradiction": 0.03,
                "other": 0.01
            }
        ),
        Evidence(
            text="Một số chuyên gia dự báo GDP Việt Nam 2024 có thể đạt từ 6.5% đến 7.5%.",
            source="https://vneconomy.vn",
            rank=3,
            nli_score={
                "entailment": 0.65,
                "neutral": 0.30,
                "contradiction": 0.04,
                "other": 0.01
            }
        )
    ]
    
    # Run debate
    try:
        verdict = orchestrator.debate(
            claim=claim,
            evidences=evidences
        )
        
        # Print results
        print("\n" + "="*80)
        print("FINAL VERDICT")
        print("="*80)
        print(f"Verdict: {verdict.verdict}")
        print(f"Confidence: {verdict.confidence:.2%}")
        print(f"Rounds used: {verdict.rounds_used}")
        print(f"Early stopped: {verdict.early_stopped} ({verdict.stop_reason})")
        print(f"\nReasoning:")
        print(f"  {verdict.reasoning}")
        print(f"\nEvidence Summary:")
        print(f"  {verdict.evidence_summary}")
        print(f"\nDebator Agreements: {verdict.debator_agreements}")
        print("="*80)
        
        return verdict
        
    except Exception as e:
        logger.error(f"Error during debate: {e}", exc_info=True)
        raise


def test_controversial_claim(models_config_path: str = None, debate_config_path: str = None, environment: str = None):
    """Test với claim controversial (sẽ trigger nhiều rounds)."""
    
    logger.info("\n\n" + "="*80)
    logger.info("TEST: Controversial Claim Debate")
    logger.info("="*80)
    
    orchestrator = AdaptiveDebateOrchestrator(
        models_config_path=models_config_path,
        debate_config_path=debate_config_path,
        environment=environment
    )
    
    # Controversial claim
    claim = "Vaccine COVID-19 gây tác dụng phụ nghiêm trọng ở 50% người tiêm."
    
    # Mixed evidences (some support, some contradict)
    evidences = [
        Evidence(
            text="WHO xác nhận vaccine COVID-19 an toàn, tác dụng phụ nghiêm trọng chỉ xảy ra ở dưới 0.01% ca.",
            source="https://who.int",
            rank=1,
            nli_score={
                "entailment": 0.05,
                "neutral": 0.10,
                "contradiction": 0.83,
                "other": 0.02
            }
        ),
        Evidence(
            text="Một số trường hợp cá nhân báo cáo tác dụng phụ sau khi tiêm vaccine COVID-19.",
            source="https://example.com",
            rank=2,
            nli_score={
                "entailment": 0.35,
                "neutral": 0.50,
                "contradiction": 0.10,
                "other": 0.05
            }
        ),
        Evidence(
            text="Nghiên cứu tại New England Journal of Medicine cho thấy tỷ lệ tác dụng phụ nghiêm trọng của vaccine là 0.005%.",
            source="https://nejm.org",
            rank=3,
            nli_score={
                "entailment": 0.03,
                "neutral": 0.12,
                "contradiction": 0.82,
                "other": 0.03
            }
        )
    ]
    
    try:
        verdict = orchestrator.debate(
            claim=claim,
            evidences=evidences
        )
        
        # Print results
        print("\n" + "="*80)
        print("FINAL VERDICT")
        print("="*80)
        print(f"Verdict: {verdict.verdict}")
        print(f"Confidence: {verdict.confidence:.2%}")
        print(f"Rounds used: {verdict.rounds_used}")
        print(f"Early stopped: {verdict.early_stopped} ({verdict.stop_reason})")
        print(f"\nReasoning:")
        print(f"  {verdict.reasoning}")
        print("="*80)
        
        return verdict
        
    except Exception as e:
        logger.error(f"Error during debate: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test Debate System')
    parser.add_argument('--config', type=str, help='Path to models_config.json')
    parser.add_argument('--debate-config', type=str, help='Path to debate_config.json')
    parser.add_argument('--environment', type=str, choices=['production', 'testing'], 
                       help='Environment mode (auto-selects config if --config not specified)')
    args = parser.parse_args()
    
    logger.info("Starting Debate System Tests...")
    
    if args.config:
        logger.info(f"Using models config: {args.config}")
    if args.debate_config:
        logger.info(f"Using debate config: {args.debate_config}")
    if args.environment:
        logger.info(f"Using environment: {args.environment}")
    
    # Test 1: Simple claim (should resolve quickly - Round 1)
    logger.info("\n" + "#"*80)
    logger.info("# TEST 1: Simple Claim (Expected: Early stop Round 1)")
    logger.info("#"*80)
    
    verdict1 = test_single_claim(
        models_config_path=args.config,
        debate_config_path=args.debate_config,
        environment=args.environment
    )
    
    # Test 2: Controversial claim (should need more rounds)
    logger.info("\n" + "#"*80)
    logger.info("# TEST 2: Controversial Claim (Expected: Multiple rounds)")
    logger.info("#"*80)
    
    verdict2 = test_controversial_claim(
        models_config_path=args.config,
        debate_config_path=args.debate_config,
        environment=args.environment
    )
    
    # Summary
    logger.info("\n\n" + "="*80)
    logger.info("TESTS COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"Test 1: {verdict1.rounds_used} rounds, early_stop={verdict1.early_stopped}")
    logger.info(f"Test 2: {verdict2.rounds_used} rounds, early_stop={verdict2.early_stopped}")
    logger.info("="*80)
