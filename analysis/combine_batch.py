#!/usr/bin/env python3
"""
Combine Batch Files for Analysis

Combines multiple metadata JSONL files and response CSV files from different
batches into single files.
Since batches are mutually exclusive (no overlapping questions), they can be
simply stacked together.

Usage:
    # Combine metadata files
    python analysis/combine_batch.py \
        --metadata data/corruption/batch1.jsonl data/corruption/batch2.jsonl data/corruption/batch3.jsonl \
        --output-metadata data/corruption/combined_metadata.jsonl

    # Combine response files
    python analysis/combine_batch.py \
        --responses data/model/model1_batch1.csv data/model/model1_batch2.csv data/model/model1_batch3.csv \
        --output-responses data/model/model1_combined.csv

    # Combine both at once
    python analysis/combine_batch.py \
        --metadata data/corruption/batch1.jsonl data/corruption/batch2.jsonl data/corruption/batch3.jsonl \
        --output-metadata combined_metadata.jsonl \
        --responses data/model/model_batch1.csv data/model/model_batch2.csv data/model/model_batch3.csv \
        --output-responses data/model/model_combined.csv

"""

import json
import csv
import argparse
from pathlib import Path


def combine_jsonl_files(input_paths, output_path):
    """
    Combine multiple JSONL files into one.
    
    Args:
        input_paths: List of Path objects to input JSONL files
        output_path: Path object for output JSONL file
    """
    total_lines = 0
    uuids_seen = set()
    duplicates = 0
    
    with output_path.open("w", encoding="utf-8") as out_f:
        for input_path in input_paths:
            print(f"  Reading {input_path}...")
            lines_from_file = 0
            
            with input_path.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse to check for duplicates
                    try:
                        entry = json.loads(line)
                        uuid = entry.get("uuid")
                        
                        if uuid and uuid in uuids_seen:
                            duplicates += 1
                            print(f"    WARNING: Duplicate UUID found: {uuid}")
                            continue
                        
                        if uuid:
                            uuids_seen.add(uuid)
                    except json.JSONDecodeError:
                        print("    WARNING: Invalid JSON line skipped")
                        continue
                    
                    # Write to output
                    out_f.write(line + "\n")
                    lines_from_file += 1
            
            total_lines += lines_from_file
            print(f"    Added {lines_from_file} entries")
    
    print(f"\nTotal entries written: {total_lines}")
    print(f"Unique UUIDs: {len(uuids_seen)}")
    if duplicates > 0:
        print(f"WARNING: {duplicates} duplicate entries were skipped")
    
    return total_lines, len(uuids_seen)


def combine_csv_files(input_paths, output_path):
    """
    Combine multiple CSV files into one.
    
    Assumes all CSVs have the same header structure.
    
    Args:
        input_paths: List of Path objects to input CSV files
        output_path: Path object for output CSV file
    """
    total_rows = 0
    uuids_seen = set()
    duplicates = 0
    header_written = False
    header = None
    
    with output_path.open("w", encoding="utf-8", newline='') as out_f:
        writer = None
        
        for input_path in input_paths:
            print(f"  Reading {input_path}...")
            rows_from_file = 0
            
            with input_path.open("r", encoding="utf-8") as in_f:
                reader = csv.DictReader(in_f)
                
                # init writer with first file's header
                if writer is None:
                    header = reader.fieldnames
                    writer = csv.DictWriter(out_f, fieldnames=header)
                    writer.writeheader()
                    header_written = True
                else:
                    # Verify header matches
                    if reader.fieldnames != header:
                        print(f"    WARNING: Header mismatch in {input_path}")
                        print(f"      Expected: {header}")
                        print(f"      Got: {reader.fieldnames}")
                
                for row in reader:
                    uuid = row.get("uuid", "")
                    
                    # check for dupes
                    if uuid and uuid in uuids_seen:
                        duplicates += 1
                        print(f"    WARNING: Duplicate UUID found: {uuid}")
                        continue
                    
                    if uuid:
                        uuids_seen.add(uuid)
                    
                    # Write row
                    writer.writerow(row)
                    rows_from_file += 1
            
            total_rows += rows_from_file
            print(f"    Added {rows_from_file} rows")
    
    print(f"\nTotal rows written: {total_rows}")
    print(f"Unique UUIDs: {len(uuids_seen)}")
    if duplicates > 0:
        print(f"WARNING: {duplicates} duplicate entries were skipped")
    
    return total_rows, len(uuids_seen)


def validate_batch_count(n_batch, file_type):
    """Validate minimum batch count.
    DEPRECATED: Kept for backward compatibility, set to < 1 always pass.
    """
    if n_batch < 1:
        print(f"\nERROR: Only {n_batch} {file_type} file(s) provided.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Combine batch files for bootstrap analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  # combine metadata from 5 batch
  python combine_batch.py \\
      --metadata batch1.jsonl batch2.jsonl batch3.jsonl batch4.jsonl batch5.jsonl \\
      --output-metadata combined_metadata.jsonl

  # combine responses from 5 batch
  python combine_batch.py \\
      --responses claude_batch1.csv claude_batch2.csv claude_batch3.csv \\
                  claude_batch4.csv claude_batch5.csv \\
      --output-responses claude_combined.csv
        """
    )
    
    parser.add_argument(
        "--metadata",
        nargs="+",
        type=Path,
        help="Input metadata JSONL files (one per batch)"
    )
    
    parser.add_argument(
        "--output-metadata",
        type=Path,
        help="Output combined metadata JSONL file"
    )
    
    parser.add_argument(
        "--responses",
        nargs="+",
        type=Path,
        help="Input response CSV files (one per batch)"
    )
    
    parser.add_argument(
        "--output-responses",
        type=Path,
        help="Output combined responses CSV file"
    )
    
    args = parser.parse_args()
    
    # validate inputs
    if not args.metadata and not args.responses:
        parser.error("Must provide either --metadata or --responses (or both)")
    
    if args.metadata and not args.output_metadata:
        parser.error("--metadata requires --output-metadata")
    
    if args.responses and not args.output_responses:
        parser.error("--responses requires --output-responses")
    
    success = True
    
    # Combine metadata files
    if args.metadata:
        print("\n" + "=" * 80)
        print("COMBINING METADATA FILES")
        print("=" * 80)
        
        # val batch count
        if not validate_batch_count(len(args.metadata), "metadata"):
            success = False
        else:
            # check all input files exist
            for path in args.metadata:
                if not path.exists():
                    print(f"ERROR: File not found: {path}")
                    success = False
            
            if success:
                total, unique = combine_jsonl_files(args.metadata, args.output_metadata)
                print(f"\n✓ Combined metadata saved to: {args.output_metadata}")
                
                # verify mutually exclusive
                if total != unique:
                    print(f"\nWARNING: Expected {total} unique entries but got {unique}")
    
    # Combine response files
    if args.responses:
        print("\n" + "=" * 80)
        print("COMBINING RESPONSE FILES")
        print("=" * 80)
        
        # val batch count
        if not validate_batch_count(len(args.responses), "response"):
            success = False
        else:
            # Check all input files exist
            for path in args.responses:
                if not path.exists():
                    print(f"ERROR: File not found: {path}")
                    success = False
            
            if success:
                total, unique = combine_csv_files(args.responses, args.output_responses)
                print(f"\nCombined responses saved to: {args.output_responses}")
                
                # Verify mutually exclusive
                if total != unique:
                    print(f"\nWARNING: Expected {total} unique entries but got {unique}")
    
    if success:
        print("\n" + "=" * 80)
        print("SUCCESS - All files combined")
        print("=" * 80)        
        return 0
    else:
        print("\n" + "=" * 80)
        print("ERRORS ENCOUNTERED - Check messages above")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit(main())