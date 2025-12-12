#!/usr/bin/env python
"""
Script to remove claim_highlight and evidence_highlight fields from all XAI data in results JSON files.
This is a one-time migration script after the XAI schema change (Dec 2025).

Usage:
    python tools/remove_highlight_fields.py [--dry-run]
"""

import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent


def remove_highlights_from_dict(d: dict) -> int:
    """
    Recursively remove claim_highlight and evidence_highlight from a dict.
    Returns count of fields removed.
    """
    removed = 0
    keys_to_remove = []
    
    for key, value in d.items():
        if key in ("claim_highlight", "evidence_highlight"):
            keys_to_remove.append(key)
        elif isinstance(value, dict):
            removed += remove_highlights_from_dict(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    removed += remove_highlights_from_dict(item)
    
    for key in keys_to_remove:
        del d[key]
        removed += 1
    
    return removed


def process_file(file_path: Path, dry_run: bool = False) -> dict:
    """Process a single JSON file and remove highlight fields."""
    print(f"\n📄 Processing: {file_path.name}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    removed_count = remove_highlights_from_dict(data)
    
    if removed_count == 0:
        print(f"   ⏭️  No highlight fields found")
        return {"file": str(file_path), "removed": 0, "status": "skipped"}
    
    print(f"   🔍 Found {removed_count} highlight fields to remove")
    
    if dry_run:
        print(f"   🔸 DRY RUN - no changes made")
        return {"file": str(file_path), "removed": removed_count, "status": "dry_run"}
    
    # Backup original
    backup_path = file_path.with_suffix(f'.backup_pre_highlight_removal_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"   💾 Backup: {backup_path.name}")
    
    # Write cleaned data back
    # Re-read original to get fresh data for backup
    with open(file_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(original_data, f, ensure_ascii=False, indent=2)
    
    # Now write cleaned data
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ Removed {removed_count} highlight fields")
    return {"file": str(file_path), "removed": removed_count, "status": "cleaned"}


def main():
    dry_run = "--dry-run" in sys.argv
    
    print("=" * 60)
    print("🧹 REMOVE HIGHLIGHT FIELDS FROM XAI DATA")
    print("=" * 60)
    
    if dry_run:
        print("⚠️  DRY RUN MODE - no files will be modified\n")
    
    # Find all results JSON files
    results_dir = PROJECT_ROOT / "results" / "vifactcheck"
    
    target_files = []
    
    # Main results files
    for pattern in ["**/vifactcheck_*_results.json", "**/phobert_xai_*.json", "**/xai_*.json"]:
        target_files.extend(results_dir.glob(pattern))
    
    # Also check root results folder
    root_results = PROJECT_ROOT / "results"
    for f in root_results.glob("*.json"):
        if "xai" in f.name.lower() or "phobert" in f.name.lower():
            target_files.append(f)
    
    # Deduplicate and filter out backups
    target_files = list(set(target_files))
    target_files = [f for f in target_files if ".backup" not in f.name]
    
    print(f"📁 Found {len(target_files)} target files\n")
    
    stats = {
        "total_files": len(target_files),
        "files_cleaned": 0,
        "files_skipped": 0,
        "total_fields_removed": 0
    }
    
    for file_path in sorted(target_files):
        result = process_file(file_path, dry_run)
        
        if result["status"] == "cleaned":
            stats["files_cleaned"] += 1
            stats["total_fields_removed"] += result["removed"]
        elif result["status"] == "skipped":
            stats["files_skipped"] += 1
        elif result["status"] == "dry_run":
            stats["total_fields_removed"] += result["removed"]
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Total files processed: {stats['total_files']}")
    print(f"Files cleaned: {stats['files_cleaned']}")
    print(f"Files skipped (no highlights): {stats['files_skipped']}")
    print(f"Total highlight fields removed: {stats['total_fields_removed']}")
    
    if dry_run:
        print("\n⚠️  This was a DRY RUN. Run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
