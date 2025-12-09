#!/usr/bin/env python3
"""
Shared data loading utilities for analysis scripts.

This module provides consistent data loading and preprocessing across
gen_*.py analysis scripts.

All scripts should use these functions to ensure they:
1. Normalize answers the same way
2. Filter invalid responses consistently
3. Count questions identically
"""

import json
import csv
import re
import pandas as pd


def normalize_category(raw):
    """Normalize sense category to standard format."""
    if raw is None:
        return ""
    
    # remove hyphens, spaces, lowercase
    s = str(raw).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    
    # map common variations
    if "common" in s and "sense" in s and "anti" not in s:
        return "commonsense"
    elif "anti" in s and "common" in s:
        return "anticommonsense"
    elif "non" in s and "sense" in s:
        return "nonsense"
    
    return s


def normalize_answer(raw):
    """
    Normalize answer to 'yes' or 'no'.
    Returns empty string if invalid.
    
    Extracts Yes/No from the START or END of the response text.
    Recognizes: Yes/No, True/False, 1/0
    
    This is the canonical answer extraction function used by all analysis scripts.
    Model runner scripts save raw responses; normalization happens here at analysis time.
    """
    if raw is None:
        return ""
    
    # Handle pandas NaN
    if isinstance(raw, float) and pd.isna(raw):
        return ""
    
    s = str(raw).strip()
    
    # Skip error responses
    if s.startswith("[ERROR]") or s.startswith("[NO_PROMPT]"):
        return ""
    
    lower = s.lower()
    
    # Check START of response first
    if lower.startswith("yes"):
        return "yes"
    if lower.startswith("no"):
        return "no"
    
    # Check END: grab last 2 lines
    lines = [line.strip() for line in s.split('\n') if line.strip()]
    if not lines:
        return ""
    
    lines_to_check = lines[-2:] if len(lines) >= 2 else lines[-1:]
    
    for line in reversed(lines_to_check):
        # strip markdown formatting and punctuation
        cleaned = line.strip('*').strip().rstrip('.!?').strip()
        
        # must be standalone boolean only
        if re.match(r'^(yes|no|true|false|1|0)$', cleaned, re.IGNORECASE):
            lower_cleaned = cleaned.lower()
            if lower_cleaned in ['yes', 'true', '1']:
                return "yes"
            elif lower_cleaned in ['no', 'false', '0']:
                return "no"
    
    return ""  # No clear answer found


def load_metadata_dict(jsonl_path):
    """
    Load metadata from JSONL file.
    Returns dict keyed by uuid with normalized ground truth.
    """
    metadata = {}
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                entry = json.loads(line)
                uuid = entry.get('uuid')
                
                if not uuid:
                    continue
                
                # Normalize ground truth using same function
                ground_truth = normalize_answer(entry.get('cladder_ground_truth'))
                
                metadata[uuid] = {
                    'question_id': entry.get('cladder_question_id'),
                    'ground_truth': ground_truth,
                    'category': entry.get('cladder_category', ''),
                    'rung': entry.get('cladder_rung'),
                    'prompt_type': entry.get('prompt_type'),
                    'corruption_type': entry.get('graph', {}).get('corruption_type') if entry.get('graph') else None
                }
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  Warning: skipping metadata line ({e})")
    
    return metadata


def load_responses_dataframe(csv_path, metadata=None):
    """
    Load responses as pandas DataFrame with normalized answers.
    Optionally filters out invalid responses if metadata provided.
    
    Args:
        csv_path: Path to CSV file
        metadata: Optional metadata dict to filter against
        
    Returns:
        DataFrame with normalized 'response' column and 'is_valid' flag
        
    Usage pattern for filtering:
        df = load_responses_dataframe(csv_path, metadata)
        df_orig = df.copy()
        df = df[df['is_valid']].copy()
        invalid_count = len(df_orig) - len(df)
    """
    df = pd.read_csv(csv_path)
    
    # Normalize responses
    df['response'] = df['response'].apply(normalize_answer)
    
    # Mark valid responses
    df['is_valid'] = df['response'] != ''
    
    # If metadata provided, also check ground truth is valid
    if metadata is not None:
        df['ground_truth'] = df['uuid'].map(lambda x: metadata.get(x, {}).get('ground_truth', ''))
        df['is_valid'] = df['is_valid'] & (df['ground_truth'] != '')
    
    return df


def load_responses_dict(csv_path):
    """
    Load responses as dict keyed by uuid with normalized answers.
    Returns:
        Dict mapping uuid -> normalized response string
    """
    responses = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            uuid = row.get('uuid')
            if uuid:
                # Normalize response
                responses[uuid] = normalize_answer(row.get('response', ''))
    
    return responses


def filter_valid_data(responses, metadata):
    """
    Filter to only include UUIDs with valid responses and ground truth.
    Args:
        responses: Dict of uuid -> response
        metadata: Dict of uuid -> metadata
        
    Returns:
        Filtered dict with only valid entries
    """
    filtered = {}
    
    for uuid, response in responses.items():
        # Must have metadata
        if uuid not in metadata:
            continue
        
        # Response must be valid (yes/no)
        if response == '':
            continue
        
        # Ground truth must be valid
        ground_truth = metadata[uuid].get('ground_truth', '')
        if ground_truth == '':
            continue
        
        filtered[uuid] = response
    
    return filtered


def get_condition(prompt_type, corruption_type):
    """
    Map prompt_type + corruption_type to experimental condition.
    """
    if prompt_type == 'no_graph':
        return 'no_graph'
    elif prompt_type == 'original_graph':
        return 'original_graph'
    elif prompt_type == 'corrupted_graph' and corruption_type:
        return corruption_type
    return 'unknown'


def get_unique_questions(data):
    """Get list of unique question IDs from data."""
    questions = set()
    for item in data.values():
        qid = item['metadata']['question_id']
        if qid:
            questions.add(qid)
    return list(questions)


def print_data_summary(metadata, responses, filtered_responses=None):
    """
    Print summary of loaded data for debugging.
    
    Args:
        metadata: Metadata dict
        responses: All responses dict/df
        filtered_responses: Responses after filtering (optional)
    """
    print(f"  Metadata entries: {len(metadata)}")
    
    if isinstance(responses, pd.DataFrame):
        print(f"  CSV rows: {len(responses)}")
        if 'is_valid' in responses.columns:
            valid_count = responses['is_valid'].sum()
            print(f"  Valid responses: {valid_count} ({valid_count/len(responses)*100:.1f}%)")
    else:
        print(f"  CSV responses: {len(responses)}")
    
    if filtered_responses is not None:
        print(f"  After filtering: {len(filtered_responses)}")


# Constants used across all scripts
CORRUPTION_TYPES = ['reverse_random_edge', 'add_collider', 'add_confounder', 'add_mediator']
CONDITIONS = ['no_graph', 'original_graph'] + CORRUPTION_TYPES
SCENARIOS = ['commonsense', 'anticommonsense', 'nonsense']
RUNGS = [1, 2, 3]
RUNG_NAMES = {1: "Association", 2: "Intervention", 3: "Counterfactual"}
SENSE_CATEGORIES = ['commonsense', 'anticommonsense', 'nonsense']

# Pattern definitions
PATTERN_NAMES = {
    "000": "all_wrong",
    "001": "only_corrupted_correct",
    "010": "only_original_correct",
    "011": "original_and_corrupted_correct",
    "100": "only_no_graph_correct",
    "101": "no_graph_and_corrupted_correct",
    "110": "no_graph_and_original_correct",
    "111": "all_correct",
}


if __name__ == '__main__':
    print("This is a shared module.")