"""
Post-process XAI with LLM API Call + PhoBERT XAI

This script generates XAI for existing results:
- SLOW PATH (debate_result != null): Call LLM API for Judge XAI
- FAST PATH (debate_result == null): Use PhoBERT XAI module

Key features:
- Verdict/confidence/reasoning remain UNCHANGED
- Resume support (skip samples that already have XAI)
- PhoBERT XAI for hybrid mode fast path samples

Author: Lockdown
Date: Dec 11, 2025
"""

import json
import shutil
import asyncio
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import argparse
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
import sys
# File is at src/pipeline/debate/postprocess_xai_with_llm.py -> go up 4 levels to project root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment
load_dotenv(project_root / '.env')

# Import LLM client
from src.pipeline.debate.llm_client import LLMClient


class PhoBERTXAIGenerator:
    """Generate XAI for fast path using PhoBERT XAI module."""
    
    def __init__(self, model_path: str = None):
        """Initialize PhoBERT XAI generator."""
        self.xai_module = None
        self.model_path = model_path or str(project_root / "results/fact_checking/pyvi/checkpoints/best_model_pyvi.pt")
        
    def load_model(self):
        """Load PhoBERT model (lazy loading)."""
        if self.xai_module is not None:
            return
            
        logger.info("Loading PhoBERT model for XAI...")
        try:
            from src.pipeline.fact_checking.xai_phobert import load_xai_model
            self.xai_module = load_xai_model(self.model_path)
            logger.info("✅ PhoBERT XAI model loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load PhoBERT model: {e}")
            raise
    
    def generate_xai(
        self,
        claim: str,
        evidence: str,
        model_verdict: str
    ) -> Dict[str, Any]:
        """
        Generate XAI for a fast path sample using PhoBERT.
        
        Args:
            claim: Statement text
            evidence: Evidence text
            model_verdict: PhoBERT's verdict (Support/Refute/NEI)
            
        Returns:
            XAI dictionary
        """
        if self.xai_module is None:
            self.load_model()
        
        # Normalize verdict
        verdict_map = {
            "Support": "SUPPORTS",
            "Refute": "REFUTES",
            "NOT_ENOUGH_INFO": "NEI",
            "NEI": "NEI"
        }
        normalized_verdict = verdict_map.get(model_verdict, "NEI")
        
        # Generate XAI using PhoBERT module
        xai_result = self.xai_module.generate_xai(
            claim=claim.replace("_", " "),  # Remove word segmentation underscores
            evidence=evidence.replace("_", " "),
            model_verdict=normalized_verdict
        )
        
        return {
            "relationship": xai_result.get("relationship", "NEI"),
            "natural_explanation": xai_result.get("natural_explanation", ""),
            "conflict_claim": xai_result.get("claim_conflict_word", ""),
            "conflict_evidence": xai_result.get("evidence_conflict_word", ""),
            "source": "FAST_PATH",
            "generated_by": "phobert_xai"
        }


class XAIGenerator:
    """Generate XAI by calling LLM API."""
    
    # Default API config (OpenRouter)
    DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"
    DEFAULT_BASE_URL_ENV = "OPENROUTER_BASE_URL"
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    
    # XAI-only prompt template (synced with judge.py wording)
    XAI_PROMPT_TEMPLATE = """**TASK:** Generate XAI (Explainable AI) fields for a fact-checking verdict.

**CONTEXT:**
- Claim: "{claim}"
- Evidence: "{evidence}"
- Verdict: {verdict}
- Reasoning: "{reasoning}"

**YOUR JOB:** Based on the verdict and reasoning above, generate structured XAI fields.
Do NOT change or question the verdict. Just explain it clearly.

**OUTPUT (JSON only, no markdown):**
{{
    "conflict_claim": "Nếu REFUTED: từ/cụm trong claim gây mâu thuẫn. Để trống '' nếu SUPPORTED/NEI",
    "conflict_evidence": "Nếu REFUTED: từ/cụm trong evidence mâu thuẫn với claim. Để trống '' nếu SUPPORTED/NEI",
    "natural_explanation_vi": "Giải thích 1-2 câu ngắn gọn bằng tiếng Việt theo format dưới đây"
}}

**RULES:**
- conflict_claim/conflict_evidence: CHỈ điền nếu verdict là REFUTED, để trống "" nếu SUPPORTED/NEI
- conflict_claim: phải là từ/cụm từ XUẤT HIỆN NGUYÊN VĂN trong tuyên bố (verbatim substring)
- conflict_evidence: phải là từ/cụm từ XUẤT HIỆN NGUYÊN VĂN trong bằng chứng (verbatim substring)
- natural_explanation_vi: Giải thích theo format:
  - SUPPORTED: "Bằng chứng cung cấp thông tin phù hợp với tuyên bố."
  - REFUTED: "Tuyên bố nói '[conflict_claim]' nhưng Bằng chứng nói '[conflict_evidence]'. Hai thông tin này mâu thuẫn."
  - NEI: "Hiện tại, bằng chứng được cung cấp chưa đủ để kết luận tuyên bố đúng hay sai."

**OUTPUT JSON:**"""

    def __init__(self, model_name: str = "deepseek/deepseek-chat-v3-0324"):
        """Initialize with LLM client."""
        self.llm_client = LLMClient()
        self.model_name = model_name
        
        # Load API credentials from environment
        self.api_key = os.getenv(self.DEFAULT_API_KEY_ENV, "")
        self.base_url = os.getenv(self.DEFAULT_BASE_URL_ENV, self.DEFAULT_BASE_URL)
        
        if not self.api_key:
            raise ValueError(f"Missing API key. Set {self.DEFAULT_API_KEY_ENV} in .env file")
        
        logger.info(f"Using model: {model_name}")
        logger.info(f"API base URL: {self.base_url}")
        
        # Stats
        self.stats = {
            "total_samples": 0,
            "samples_with_debate": 0,
            "xai_generated": 0,
            "xai_failed": 0,
            "skipped_no_debate": 0
        }
    
    async def generate_xai_for_sample(
        self,
        claim: str,
        evidence: str,
        verdict: str,
        reasoning: str
    ) -> Optional[Dict[str, Any]]:
        """
        Call LLM to generate XAI for a single sample.
        
        Returns:
            XAI dict or None if failed
        """
        # Clean text
        claim_clean = claim.replace("_", " ").strip()
        evidence_clean = evidence.replace("_", " ").strip()
        reasoning_clean = reasoning[:500] if reasoning else ""  # Limit reasoning length
        
        prompt = self.XAI_PROMPT_TEMPLATE.format(
            claim=claim_clean,
            evidence=evidence_clean,
            verdict=verdict,
            reasoning=reasoning_clean
        )
        
        try:
            response = await self.llm_client.generate_async(
                model=self.model_name,
                prompt=prompt,
                api_key=self.api_key,
                base_url=self.base_url,
                max_tokens=500,
                temperature=0.3  # Low temperature for consistency
            )
            
            # Parse JSON response
            import re
            # Clean response
            cleaned = re.sub(r'```(?:json)?', '', response, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r'```', '', cleaned).strip()
            
            # Fix invalid control characters
            cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', cleaned)
            
            # Try to find and parse JSON object
            xai_data = None
            
            # Method 1: Try parsing the whole response as JSON
            try:
                # Find first { and last }
                start = cleaned.find('{')
                end = cleaned.rfind('}')
                if start != -1 and end != -1:
                    json_str = cleaned[start:end+1]
                    xai_data = json.loads(json_str)
            except json.JSONDecodeError:
                pass
            
            # Method 2: Try with simple regex if method 1 failed
            if xai_data is None:
                json_match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
                if json_match:
                    try:
                        xai_data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass
            
            if xai_data:
                # Normalize verdict for relationship field
                verdict_map = {
                    "SUPPORTED": "SUPPORTS",
                    "SUPPORTS": "SUPPORTS",
                    "REFUTED": "REFUTES", 
                    "REFUTES": "REFUTES",
                    "NEI": "NEI",
                    "NOT_ENOUGH_INFO": "NEI"
                }
                relationship = verdict_map.get(verdict.upper(), "NEI") if verdict else "NEI"
                
                return {
                    "relationship": relationship,
                    "natural_explanation": xai_data.get("natural_explanation_vi", ""),
                    "conflict_claim": xai_data.get("conflict_claim", ""),
                    "conflict_evidence": xai_data.get("conflict_evidence", ""),
                    "source": "SLOW_PATH",
                    "generated_by": "llm_postprocess"
                }
            else:
                logger.warning(f"No JSON found in response: {cleaned[:100]}...")
                return None
                
        except Exception as e:
            logger.error(f"Error generating XAI: {e}")
            return None
    
    async def process_file(
        self,
        input_path: Path,
        backup: bool = True,
        concurrency: int = 10,
        phobert_generator: Optional['PhoBERTXAIGenerator'] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Process a single results file and add XAI.
        - Slow path (debate_result): LLM API
        - Fast path (no debate_result): PhoBERT XAI (if generator provided)
        
        Args:
            input_path: Path to the results JSON file
            backup: Whether to backup original file
            concurrency: Number of concurrent API calls (semaphore limit)
            phobert_generator: PhoBERT XAI generator for fast path (optional)
            
        Returns:
            Statistics about the processing
        """
        logger.info(f"Processing: {input_path}")
        
        # Load original data
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Backup original file
        if backup:
            backup_path = input_path.with_suffix(f'.backup_llm_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            shutil.copy(input_path, backup_path)
            logger.info(f"✅ Backup created: {backup_path.name}")
        
        # Store original metrics for verification
        original_metrics = {
            "total_samples": data.get("total_samples"),
            "model_accuracy": data.get("model_accuracy"),
            "final_accuracy": data.get("final_accuracy")
        }
        
        # Find samples that need XAI (skip those already processed)
        results = data.get("results", [])
        self.stats["total_samples"] = len(results)
        
        slow_path_samples = []  # Has debate_result -> LLM XAI
        fast_path_samples = []  # No debate_result -> PhoBERT XAI
        already_has_slow_xai = 0
        already_has_fast_xai = 0
        
        for i, result in enumerate(results):
            if result.get("debate_result") is not None:
                self.stats["samples_with_debate"] += 1
                # Skip if already has XAI (resume support) unless --force
                if result["debate_result"].get("xai") and not force:
                    already_has_slow_xai += 1
                else:
                    slow_path_samples.append((i, result))
            else:
                self.stats["skipped_no_debate"] += 1
                # Fast path: check if PhoBERT XAI needed
                if phobert_generator:
                    if result.get("phobert_xai") and not force:
                        already_has_fast_xai += 1
                    else:
                        fast_path_samples.append((i, result))
        
        if already_has_slow_xai > 0:
            logger.info(f"⏭️  Skipping {already_has_slow_xai} slow path samples (already have XAI)")
        if already_has_fast_xai > 0:
            logger.info(f"⏭️  Skipping {already_has_fast_xai} fast path samples (already have XAI)")
        
        # === Process Fast Path (PhoBERT XAI) ===
        if phobert_generator and fast_path_samples:
            logger.info(f"📌 Processing {len(fast_path_samples)} FAST PATH samples (PhoBERT XAI)...")
            for idx, result in fast_path_samples:
                try:
                    xai = phobert_generator.generate_xai(
                        claim=result.get("statement", ""),
                        evidence=result.get("evidence", ""),
                        model_verdict=result.get("model_verdict", "NEI")
                    )
                    results[idx]["phobert_xai"] = xai
                    self.stats["xai_generated"] += 1
                except Exception as e:
                    logger.warning(f"Fast path sample {idx} failed: {e}")
                    self.stats["xai_failed"] += 1
                
                # Progress log every 100 samples
                processed = fast_path_samples.index((idx, result)) + 1
                if processed % 100 == 0 or processed == len(fast_path_samples):
                    logger.info(f"Fast path progress: {processed}/{len(fast_path_samples)}")
        
        # === Process Slow Path (LLM XAI) ===
        logger.info(f"📌 Processing {len(slow_path_samples)} SLOW PATH samples (LLM XAI)...")
        logger.info(f"🚀 Using {concurrency} concurrent workers (semaphore)")
        
        # Semaphore-based concurrent processing (like eval_vifactcheck_pipeline.py)
        if slow_path_samples:
            sem = asyncio.Semaphore(concurrency)
            processed_count = 0
            
            async def bounded_generate_xai(idx: int, result: dict):
                """Generate XAI with semaphore limit."""
                async with sem:
                    try:
                        xai = await self.generate_xai_for_sample(
                            claim=result.get("statement", ""),
                            evidence=result.get("evidence", ""),
                            verdict=result["debate_result"].get("verdict", "NEI"),
                            reasoning=result["debate_result"].get("reasoning", "")
                        )
                        return (idx, xai, None)
                    except Exception as e:
                        return (idx, None, str(e))
            
            # Create all tasks
            tasks = [bounded_generate_xai(idx, result) for idx, result in slow_path_samples]
            
            # Process with progress tracking using as_completed
            for coro in asyncio.as_completed(tasks):
                idx, xai, error = await coro
                processed_count += 1
                
                if xai:
                    results[idx]["debate_result"]["xai"] = xai
                    self.stats["xai_generated"] += 1
                else:
                    if error:
                        logger.warning(f"Sample {idx} failed: {error}")
                    self.stats["xai_failed"] += 1
                
                # Progress log every 20 samples
                if processed_count % 20 == 0 or processed_count == len(slow_path_samples):
                    logger.info(f"Slow path progress: {processed_count}/{len(slow_path_samples)} samples processed")
        else:
            logger.info("No slow path samples to process")
        
        # Verify metrics unchanged
        new_metrics = {
            "total_samples": data.get("total_samples"),
            "model_accuracy": data.get("model_accuracy"),
            "final_accuracy": data.get("final_accuracy")
        }
        
        if original_metrics != new_metrics:
            logger.error("❌ CRITICAL: Metrics changed! Aborting save.")
            logger.error(f"Original: {original_metrics}")
            logger.error(f"New: {new_metrics}")
            raise ValueError("Metrics integrity check failed")
        
        # Save updated data
        with open(input_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Saved: {input_path}")
        logger.info(f"   Stats: {self.stats}")
        
        return self.stats
    
    async def close(self):
        """Close LLM client."""
        if self.llm_client:
            await self.llm_client.close()


async def main_async(args):
    """Async main function."""
    results_dir = project_root / "results" / "vifactcheck"
    
    # Determine which files to process
    files_to_process = []
    is_hybrid_file = {}  # Track which files are hybrid (need PhoBERT XAI)
    
    for split in args.splits:
        split_dir = results_dir / split
        
        if args.mode in ["hybrid", "both"]:
            hybrid_file = split_dir / "hybrid_debate" / f"vifactcheck_{split}_results.json"
            if hybrid_file.exists():
                files_to_process.append(hybrid_file)
                is_hybrid_file[str(hybrid_file)] = True
        
        if args.mode in ["full", "both"]:
            full_file = split_dir / "full_debate" / f"vifactcheck_{split}_results.json"
            if full_file.exists():
                files_to_process.append(full_file)
                is_hybrid_file[str(full_file)] = False
    
    if not files_to_process:
        logger.error("No result files found!")
        return
    
    logger.info(f"Found {len(files_to_process)} files to process:")
    for f in files_to_process:
        file_type = "hybrid" if is_hybrid_file.get(str(f)) else "full"
        logger.info(f"  - {f.relative_to(results_dir)} ({file_type})")
    
    if args.dry_run:
        logger.info("DRY RUN - No changes will be made")
        return
    
    # Initialize PhoBERT XAI generator if needed
    phobert_generator = None
    if args.include_phobert and any(is_hybrid_file.values()):
        logger.info("📌 Initializing PhoBERT XAI generator for fast path...")
        phobert_generator = PhoBERTXAIGenerator()
        phobert_generator.load_model()
    
    # Process files
    generator = XAIGenerator(model_name=args.model)
    total_stats = {
        "files_processed": 0,
        "total_xai_generated": 0,
        "total_xai_failed": 0
    }
    
    try:
        for file_path in files_to_process:
            try:
                # Use PhoBERT generator only for hybrid files
                use_phobert = phobert_generator if is_hybrid_file.get(str(file_path)) else None
                
                stats = await generator.process_file(
                    file_path,
                    backup=not args.no_backup,
                    concurrency=args.concurrency,
                    phobert_generator=use_phobert,
                    force=args.force
                )
                total_stats["files_processed"] += 1
                total_stats["total_xai_generated"] += stats["xai_generated"]
                total_stats["total_xai_failed"] += stats["xai_failed"]
                
                # Reset generator stats for next file
                generator.stats = {
                    "total_samples": 0,
                    "samples_with_debate": 0,
                    "xai_generated": 0,
                    "xai_failed": 0,
                    "skipped_no_debate": 0
                }
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                raise
    finally:
        await generator.close()
    
    logger.info("=" * 60)
    logger.info("✅ POST-PROCESSING COMPLETE")
    logger.info(f"   Files processed: {total_stats['files_processed']}")
    logger.info(f"   Total XAI generated: {total_stats['total_xai_generated']}")
    logger.info(f"   Total XAI failed: {total_stats['total_xai_failed']}")
    if phobert_generator:
        logger.info("   PhoBERT XAI: Enabled for hybrid files")
    logger.info("=" * 60)
    logger.info("⚠️  IMPORTANT: Verify that accuracy metrics are UNCHANGED!")


def main():
    parser = argparse.ArgumentParser(description="Post-process XAI with LLM API")
    parser.add_argument(
        "--mode", 
        choices=["hybrid", "full", "both"],
        default="both",
        help="Which results to process"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        help="Which splits to process (default: test only)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating backup files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--model",
        default="deepseek/deepseek-chat-v3-0324",
        help="LLM model to use for XAI generation"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent API calls (semaphore limit)"
    )
    parser.add_argument(
        "--include-phobert",
        action="store_true",
        help="Include PhoBERT XAI for fast path samples in hybrid mode"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regenerate XAI even for samples that already have XAI"
    )
    
    args = parser.parse_args()
    
    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
