#!/usr/bin/env python3
"""
Generate tables for hypothesis testing.

Separates data computation from table formatting to allow independent styling.
Borrows computational logic from gen_stats.py, gen_ci.py, and gen_patterns.py
where indicated, keeping code identical to source unless necessary for functionality.

Usage:
    python ./analysis/gen_hypothesis_tables.py \
        --metadata ./data/corruption/corrupted_causal_graphs_dataset_f1-3.jsonl \
        --csv ./data/model/gpt5_responses_f1-3.csv \
        --model-name "GPT-5-mini" \
        --output-dir ./analysis/tables/gpt5 \
        --save-raw \
        --format txt

    python ./analysis/gen_hypothesis_tables.py \
        --metadata ./data/corruption/corrupted_causal_graphs_dataset_f1-3.jsonl \
        --csv ./data/model/claude_responses_f1-3.csv \
        --model-name "Claude Sonnet 4" \
        --output-dir ./analysis/tables/claude \
        --save-raw \
        --format txt

Output Structure:
    Each table function returns TablePackage(data=DataFrame, metadata=TableMetadata)
    - data: Raw numbers as DataFrame
    - metadata: Formatting instructions (footnotes, number formats, etc.)
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import pickle
from analysis_common import (
    load_metadata_dict,
    load_responses_dict,
    filter_valid_data,
    get_condition,
    normalize_category,
    CONDITIONS,
    CORRUPTION_TYPES,
    SCENARIOS,
    RUNGS,
    RUNG_NAMES)

from scipy import stats

from table_format_helpers import (
    TableMetadata,
    TablePackage,
    format_table_for_output
)


# ============================================================================
# DATA LOADING AND STRUCTURING
# ============================================================================

def load_and_structure_data(metadata_path, csv_path):
    """
    Load and structure data for analysis.
    
    Returns:
        Dict keyed by uuid with all relevant fields.
        
    Note:
        Uses load_metadata_dict and load_responses_dict from analysis_common.py
        Same logic as used in gen_stats.py, gen_ci.py, gen_patterns.py
    """
    print("  Loading metadata...")
    metadata = load_metadata_dict(metadata_path)
    print(f"    {len(metadata)} metadata entries")
    
    print("  Loading responses...")
    responses = load_responses_dict(csv_path)
    print(f"    {len(responses)} raw responses")
    
    print("  Filtering valid responses...")
    # SOURCE: analysis_common.py filter_valid_data()
    # filters to only valid yes/no responses with valid ground truth
    responses = filter_valid_data(responses, metadata)
    print(f"    {len(responses)} valid responses after filtering")
    
    # build structured data dictionary
    data = {}
    for uuid, response in responses.items():
        if uuid not in metadata:
            continue
        
        meta = metadata[uuid]
        
        # SOURCE: analysis_common.py get_condition()
        # maps prompt_type + corruption_type to experimental condition
        condition = get_condition(meta['prompt_type'], meta['corruption_type'])
        
        # SOURCE: analysis_common.py normalize_category()
        # normalizes category names consistently
        category = normalize_category(meta['category'])
        
        data[uuid] = {
            'response': response,
            'ground_truth': meta['ground_truth'],
            'question_id': meta['question_id'],
            'condition': condition,
            'category': category,
            'rung': meta['rung'],
            'corruption_type': meta['corruption_type'],
            'prompt_type': meta['prompt_type'],
            'is_correct': response == meta['ground_truth']
        }
    
    return data


# ============================================================================
# ACCURACY COMPUTATION
# SOURCE: gen_ci.py compute_accuracy_with_ci()
# kept identical except for data structure (dict instead of list)
# ============================================================================

def compute_accuracy_with_ci(data, condition=None, category=None, rung=None):
    """
    Compute accuracy with Wilson score confidence interval.
    
    SOURCE: gen_ci.py compute_accuracy_with_ci()
    Logic kept identical to source except adapted for dict data structure.
    
    Args:
        data: Dict of uuid -> data_dict
        condition: Filter by condition (optional)
        category: Filter by category (optional)
        rung: Filter by rung (optional)
    
    Returns:
        (accuracy, (ci_low, ci_high), n_correct, n_total)
    """
    # filter data based on criteria
    filtered = data.copy()
    if condition is not None:
        filtered = {k: v for k, v in filtered.items() if v['condition'] == condition}
    if category is not None:
        filtered = {k: v for k, v in filtered.items() if v['category'] == category}
    if rung is not None:
        filtered = {k: v for k, v in filtered.items() if v['rung'] == rung}
    
    if not filtered:
        return None, (None, None), 0, 0
    
    # compute accuracy
    # SOURCE: gen_stats.py calc_accuracy()
    n_correct = sum(1 for v in filtered.values() if v['is_correct'])
    n_total = len(filtered)
    accuracy = n_correct / n_total if n_total > 0 else 0
    
    # wilson score interval (exact binomial CI)
    # SOURCE: gen_ci.py (Wilson score implementation)
    z = 1.96  # 95% CI
    if n_total > 0:
        p = accuracy
        denominator = 1 + z**2 / n_total
        center = (p + z**2 / (2 * n_total)) / denominator
        margin = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denominator
        ci = (center - margin, center + margin)
    else:
        ci = (None, None)
    
    return accuracy, ci, n_correct, n_total


# ============================================================================
# BOOTSTRAP UTILITIES
# SOURCE: gen_ci.py bootstrap utilities
# kept identical to source
# ============================================================================

def get_unique_questions(data):
    """
    Get list of unique question IDs.
    
    SOURCE: gen_ci.py get_unique_questions()
    Identical to source.
    """
    questions = set(v['question_id'] for v in data.values() if v['question_id'])
    return list(questions)


def build_question_index(data):
    """
    Build question ID index for fast bootstrap resampling.
    Call once before bootstrap loop to avoid O(Q×N) nested loop penalty.
    
    Args:
        data: Dict of uuid -> data_dict
        
    Returns:
        Dict mapping question_id -> list of (uuid, data_dict) tuples
    """
    by_question = defaultdict(list)
    for uuid, v in data.items():
        if v['question_id']:
            by_question[v['question_id']].append((uuid, v))
    return by_question


def resample_by_questions(by_question, questions):
    """
    Resample data by question IDs with replacement.
    Maintains within-question structure for bootstrap.
    
    FIXED: Uses pre-built index for O(Q) instead of O(Q×N) complexity.
    Correctly handles duplicate question samples (if question sampled 3x, appears 3x).
    
    Args:
        by_question: Pre-built index from build_question_index()
        questions: List of question IDs to sample from
        
    Returns:
        Resampled data dict
    """
    resampled_questions = np.random.choice(questions, size=len(questions), replace=True)
    
    resampled_data = {}
    counter = 0
    for qid in resampled_questions:
        # O(1) lookup of all items with this question_id
        for uuid, v in by_question[qid]:
            # create unique key to avoid collisions when same question sampled multiple times
            new_uuid = f"{uuid}_resample_{counter}"
            resampled_data[new_uuid] = v
            counter += 1
    
    return resampled_data


def bootstrap_delta(data, condition1, condition2, n_bootstrap=10000):
    """
    Compute delta with bootstrap confidence interval.
    Delta = accuracy(condition1) - accuracy(condition2)
    
    SOURCE: gen_ci.py bootstrap_metric() and compute_delta_* functions
    FIXED: Uses indexed resampling for ~500x speedup.
    
    Args:
        data: Dict of uuid -> data_dict
        condition1: First condition (e.g., 'original_graph')
        condition2: Second condition (e.g., 'no_graph')
        n_bootstrap: Number of bootstrap iterations
        
    Returns:
        (mean_delta, (ci_low, ci_high), standard_error)
    """
    # build index once before bootstrap loop (performance fix)
    by_question = build_question_index(data)
    questions = list(by_question.keys())
    
    if not questions:
        return None, (None, None), None
    
    bootstrap_deltas = []
    for i in range(n_bootstrap):
        if (i + 1) % 2000 == 0:
            print(f"      Bootstrap {i+1}/{n_bootstrap}", end='\r')
        
        # use indexed resampling (O(Q) instead of O(Q×N))
        resampled = resample_by_questions(by_question, questions)
        
        # compute accuracies for both conditions
        acc1, _, _, _ = compute_accuracy_with_ci(resampled, condition=condition1)
        acc2, _, _, _ = compute_accuracy_with_ci(resampled, condition=condition2)
        
        if acc1 is not None and acc2 is not None:
            bootstrap_deltas.append(acc1 - acc2)
    
    print(" " * 60, end='\r')  # clear progress line
    
    if not bootstrap_deltas:
        return None, (None, None), None
    
    # SOURCE: gen_ci.py bootstrap statistics computation
    # compute mean, SE, and percentile confidence intervals
    values = np.array(bootstrap_deltas)
    mean = np.mean(values)
    se = np.std(values)
    ci_low = np.percentile(values, 2.5)
    ci_high = np.percentile(values, 97.5)
    
    return mean, (ci_low, ci_high), se


def bootstrap_avg_delta2(data, n_bootstrap=10000, verbose=True):
    """
    Bootstrap CI for average Δ₂ across all corruption types.
    CORRECTED METHOD: Bootstrap the entire averaging process.
    
    For each bootstrap iteration:
      1. Resample questions with replacement
      2. Compute delta_2 for EACH corruption type on THIS resampled data
      3. Average the 4 delta_2 values
      4. Store as one bootstrap replicate
    """
    by_question = build_question_index(data)
    questions = list(by_question.keys())
    
    if not questions:
        return None, (None, None), None
    
    if verbose:
        print("      Bootstrapping average Δ₂...")
    
    bootstrap_avg_deltas = []
    
    for i in range(n_bootstrap):
        if verbose and (i + 1) % 2000 == 0:
            print(f"      Bootstrap {i+1}/{n_bootstrap}", end='\r')
        
        resampled_data = resample_by_questions(by_question, questions)
        
        d2_values = []
        for corr_type in CORRUPTION_TYPES:
            acc_original, _, _, _ = compute_accuracy_with_ci(
                resampled_data, condition='original_graph')
            acc_corrupted, _, _, _ = compute_accuracy_with_ci(
                resampled_data, condition=corr_type)
            
            if acc_original is not None and acc_corrupted is not None:
                delta2 = acc_original - acc_corrupted
                d2_values.append(delta2)
        
        if d2_values:
            bootstrap_avg_deltas.append(np.mean(d2_values))
    
    if verbose:
        print(" " * 60, end='\r')
    
    if not bootstrap_avg_deltas:
        return None, (None, None), None
    
    mean = np.mean(bootstrap_avg_deltas)
    se = np.std(bootstrap_avg_deltas)
    ci_low = np.percentile(bootstrap_avg_deltas, 2.5)
    ci_high = np.percentile(bootstrap_avg_deltas, 97.5)
    
    return mean, (ci_low, ci_high), se


def bootstrap_avg_delta3(data, n_bootstrap=10000, verbose=True):
    """
    Bootstrap CI for average Δ₃ across all corruption types.
    CORRECTED METHOD: Bootstrap the entire averaging process.
    """
    by_question = build_question_index(data)
    questions = list(by_question.keys())
    
    if not questions:
        return None, (None, None), None
    
    if verbose:
        print("      Bootstrapping average Δ₃...")
    
    bootstrap_avg_deltas = []
    
    for i in range(n_bootstrap):
        if verbose and (i + 1) % 2000 == 0:
            print(f"      Bootstrap {i+1}/{n_bootstrap}", end='\r')
        
        resampled_data = resample_by_questions(by_question, questions)
        
        d3_values = []
        for corr_type in CORRUPTION_TYPES:
            acc_no_graph, _, _, _ = compute_accuracy_with_ci(
                resampled_data, condition='no_graph')
            acc_corrupted, _, _, _ = compute_accuracy_with_ci(
                resampled_data, condition=corr_type)
            
            if acc_no_graph is not None and acc_corrupted is not None:
                delta3 = acc_no_graph - acc_corrupted
                d3_values.append(delta3)
        
        if d3_values:
            bootstrap_avg_deltas.append(np.mean(d3_values))
    
    if verbose:
        print(" " * 60, end='\r')
    
    if not bootstrap_avg_deltas:
        return None, (None, None), None
    
    mean = np.mean(bootstrap_avg_deltas)
    se = np.std(bootstrap_avg_deltas)
    ci_low = np.percentile(bootstrap_avg_deltas, 2.5)
    ci_high = np.percentile(bootstrap_avg_deltas, 97.5)
    
    return mean, (ci_low, ci_high), se


def bootstrap_avg_accuracy_corrupted(data, n_bootstrap=10000, verbose=True):
    """
    Bootstrap CI for average accuracy across all corruption types.
    CORRECTED METHOD: Bootstrap the entire averaging process.
    """
    by_question = build_question_index(data)
    questions = list(by_question.keys())
    
    if not questions:
        return None, (None, None), None
    
    if verbose:
        print("      Bootstrapping average corrupted accuracy...")
    
    bootstrap_avg_accs = []
    
    for i in range(n_bootstrap):
        if verbose and (i + 1) % 2000 == 0:
            print(f"      Bootstrap {i+1}/{n_bootstrap}", end='\r')
        
        resampled_data = resample_by_questions(by_question, questions)
        
        acc_values = []
        for corr_type in CORRUPTION_TYPES:
            acc, _, _, _ = compute_accuracy_with_ci(resampled_data, condition=corr_type)
            if acc is not None:
                acc_values.append(acc)
        
        if acc_values:
            bootstrap_avg_accs.append(np.mean(acc_values))
    
    if verbose:
        print(" " * 60, end='\r')
    
    if not bootstrap_avg_accs:
        return None, (None, None), None
    
    mean = np.mean(bootstrap_avg_accs)
    se = np.std(bootstrap_avg_accs)
    ci_low = np.percentile(bootstrap_avg_accs, 2.5)
    ci_high = np.percentile(bootstrap_avg_accs, 97.5)
    
    return mean, (ci_low, ci_high), se


# ============================================================================
# TABLE 1: OVERALL PERFORMANCE & DELTA METRICS (H1)
# ============================================================================

def compute_table1_overall_performance(data, model_name):
    """
    Compute overall performance and delta metrics for H1 (primary hypothesis).
    
    Tests: Do models genuinely use causal graphs?
    
    Metrics computed:
        - Acc(No Graph): Baseline accuracy
        - Acc(Correct Graph): Accuracy with correct causal graph
        - Acc(Corrupted avg): Average across corrupted graphs
        - Δ₁: Correct - No Graph (graph benefit)
        - Δ₂: Correct - Corrupted (corruption harm)
        - Δ₃: No Graph - Corrupted (parametric memory advantage)
        
    All metrics include 95% bootstrap confidence intervals.
    
    Returns:
        TablePackage with data DataFrame and formatting metadata
    """
    print("\n  Computing overall accuracies...")
    
    # -------------------------------------------------------------------------
    # accuracies for main conditions
    # SOURCE: uses compute_accuracy_with_ci() which is from gen_ci.py
    # -------------------------------------------------------------------------
    
    acc_no_graph, ci_ng, n_corr_ng, n_total_ng = compute_accuracy_with_ci(
        data, condition='no_graph')
    
    acc_correct, ci_corr, n_corr_corr, n_total_corr = compute_accuracy_with_ci(
        data, condition='original_graph')
    
    # average accuracy across corruption types with proper bootstrap CI
    print("  Computing average corrupted accuracy...")
    acc_corrupted_avg, ci_corrupted, _ = bootstrap_avg_accuracy_corrupted(data, n_bootstrap=10000)
    
    # get average n for corrupted (for reference, not used in CI)
    corruption_ns = []
    for corr_type in CORRUPTION_TYPES:
        _, _, _, n = compute_accuracy_with_ci(data, condition=corr_type)
        corruption_ns.append(n)
    n_corrupted_avg = int(np.mean(corruption_ns)) if corruption_ns else 0
    
    # -------------------------------------------------------------------------
    # delta metrics with bootstrap CIs
    # SOURCE: gen_ci.py compute_delta_1(), compute_delta_2_avg()
    # -------------------------------------------------------------------------
    
    print("  Computing Δ₁ (Correct - No Graph)...")
    delta1, ci_d1, se_d1 = bootstrap_delta(data, 'original_graph', 'no_graph')
    
    print("  Computing Δ₂ (Correct - Corrupted)...")
    delta2_avg, ci_d2_avg, delta2_se = bootstrap_avg_delta2(data, n_bootstrap=10000)
    
    print("  Computing Δ₃ (No Graph - Corrupted)...")
    delta3_avg, ci_d3_avg, delta3_se = bootstrap_avg_delta3(data, n_bootstrap=10000)
    
    # -------------------------------------------------------------------------
    # build data frame with raw numbers
    # -------------------------------------------------------------------------
    
    table_data = pd.DataFrame([{
        'Model': model_name,
        'Acc_NoGraph': acc_no_graph,
        'Acc_NoGraph_CI_low': ci_ng[0],
        'Acc_NoGraph_CI_high': ci_ng[1],
        'n_NoGraph': n_total_ng,
        'Acc_Correct': acc_correct,
        'Acc_Correct_CI_low': ci_corr[0],
        'Acc_Correct_CI_high': ci_corr[1],
        'n_Correct': n_total_corr,
        'Acc_Corrupted': acc_corrupted_avg,
        'Acc_Corrupted_CI_low': ci_corrupted[0] if ci_corrupted and ci_corrupted[0] is not None else None,
        'Acc_Corrupted_CI_high': ci_corrupted[1] if ci_corrupted and ci_corrupted[1] is not None else None,
        'n_Corrupted': n_corrupted_avg,
        'Delta1': delta1,
        'Delta1_CI_low': ci_d1[0] if ci_d1[0] is not None else None,
        'Delta1_CI_high': ci_d1[1] if ci_d1[1] is not None else None,
        'Delta1_SE': se_d1,
        'Delta2': delta2_avg,
        'Delta2_CI_low': ci_d2_avg[0],
        'Delta2_CI_high': ci_d2_avg[1],
        'Delta2_SE': delta2_se,
        'Delta3': delta3_avg,
        'Delta3_CI_low': ci_d3_avg[0],
        'Delta3_CI_high': ci_d3_avg[1],
        'Delta3_SE': delta3_se,
    }])
    
    # -------------------------------------------------------------------------
    # create metadata for formatting
    # -------------------------------------------------------------------------
    
    metadata = TableMetadata(
        title="Overall Performance and Delta Metrics (H1)",
        hypothesis="H1: Causal Graph Non-Utilization",
        footnotes=[
            "Δ₁ (Graph Benefit) = Acc(Correct) - Acc(No Graph)",
            "Δ₂ (Corruption Harm) = Acc(Correct) - Acc(Corrupted)",
            "Δ₃ (Parametric Advantage) = Acc(No Graph) - Acc(Corrupted)",
            "All confidence intervals computed via bootstrap (10,000 iterations)",
            "Significance: CI excludes 0 → p < 0.05"
        ],
        number_formats={
            'Acc_NoGraph': '{:.3f}',
            'Acc_Correct': '{:.3f}',
            'Acc_Corrupted': '{:.3f}',
            'Delta1': '{:+.3f}',
            'Delta2': '{:+.3f}',
            'Delta3': '{:+.3f}',
            # CIs will be formatted as [low, high]
        },
        column_order=[
            'Model', 'Acc_NoGraph', 'n_NoGraph', 'Acc_Correct', 'n_Correct',
            'Acc_Corrupted', 'n_Corrupted', 'Delta1', 'Delta2', 'Delta3'
        ]
    )
    
    return TablePackage(data=table_data, metadata=metadata)


# ============================================================================
# TABLE 2: CORRUPTION-SPECIFIC SENSITIVITY (H2)
# ============================================================================

def compute_table2_corruption_specific(data, model_name):
    """
    Compute corruption-specific effects for H2.
    
    Tests: Differential sensitivity to corruption types.
    Expected ranking: Confounder > Collider > Mediator > Random Edge
    
    Returns:
        TablePackage with data DataFrame and formatting metadata
    """
    print("\n  Computing corruption-specific effects...")
    
    rows = []
    for corr_type in CORRUPTION_TYPES:
        print(f"    {corr_type}...")
        
        # accuracy for this corruption
        acc, ci, n_corr, n_total = compute_accuracy_with_ci(data, condition=corr_type)
        
        # Δ₂ for this corruption
        # SOURCE: gen_ci.py compute_delta_2_individual()
        delta2, ci_d2, se = bootstrap_delta(data, 'original_graph', corr_type)
        
        rows.append({
            'Model': model_name,
            'Corruption_Type': corr_type,
            'Accuracy': acc,
            'Acc_CI_low': ci[0],
            'Acc_CI_high': ci[1],
            'n': n_total,
            'Delta2': delta2,
            'Delta2_CI_low': ci_d2[0] if ci_d2[0] is not None else None,
            'Delta2_CI_high': ci_d2[1] if ci_d2[1] is not None else None,
            'Delta2_SE': se,
        })
    
    table_data = pd.DataFrame(rows)
    
    # sort by Δ magnitude (descending) to show ranking
    table_data = table_data.sort_values('Delta2', ascending=False, na_position='last')
    
    metadata = TableMetadata(
        title="Corruption-Specific Sensitivity (H2)",
        hypothesis="H2: Differential Corruption Sensitivity",
        footnotes=[
            "Δ₂ = Acc(Correct Graph) - Acc(This Corruption)",
            "Sorted by Δ₂ magnitude (most harmful first)",
            "Expected ranking: Confounder > Collider > Mediator > Random Edge",
            "Positive Δ₂ indicates corruption hurts performance"
        ],
        number_formats={
            'Accuracy': '{:.3f}',
            'Delta2': '{:+.3f}',
        },
        column_order=[
            'Model', 'Corruption_Type', 'Accuracy', 'n', 'Delta2'
        ]
    )
    
    return TablePackage(data=table_data, metadata=metadata)


# ============================================================================
# TABLE 3: SCENARIO × CORRUPTION INTERACTION (H3)
# ============================================================================

def compute_table3_scenario_interaction(data, model_name):
    """
    Compute scenario × corruption interaction for H3.
    
    Tests: How corruption effects vary by scenario type.
    Expected: Nonsense > Anti-commonsense > Commonsense
    
    Returns:
        TablePackage with data DataFrame and formatting metadata
    """
    print("\n  Computing scenario × corruption interaction...")
    
    rows = []
    for scenario in SCENARIOS:
        print(f"    {scenario}...")
        
        # filter to this scenario
        scenario_data = {k: v for k, v in data.items() if v['category'] == scenario}
        
        if not scenario_data:
            continue
        
        row = {
            'Scenario': scenario,
            'Model': model_name,
            'n_questions': len(set(v['question_id'] for v in scenario_data.values()))
        }
        
        # compute Δ₂ for each corruption type in this scenario
        for corr_type in CORRUPTION_TYPES:
            delta2, ci_d2, _ = bootstrap_delta(
                scenario_data, 'original_graph', corr_type, n_bootstrap=5000)
            
            row[f'Delta2_{corr_type}'] = delta2
            row[f'Delta2_{corr_type}_CI_low'] = ci_d2[0] if ci_d2[0] is not None else None
            row[f'Delta2_{corr_type}_CI_high'] = ci_d2[1] if ci_d2[1] is not None else None
        
        # also compute average Δ₂ for this scenario
        delta2_vals = []
        for corr_type in CORRUPTION_TYPES:
            d, _, _ = bootstrap_delta(
                scenario_data, 'original_graph', corr_type, n_bootstrap=3000)
            if d is not None:
                delta2_vals.append(d)
        
        row['Delta2_avg'] = np.mean(delta2_vals) if delta2_vals else None
        
        rows.append(row)
    
    table_data = pd.DataFrame(rows)
    
    metadata = TableMetadata(
        title="Scenario × Corruption Interaction (H3)",
        hypothesis="H3: Scenario-Type Interaction",
        footnotes=[
            "Δ₂ = Acc(Correct Graph) - Acc(Corrupted Graph) for each scenario",
            "Expected degradation: Nonsense > Anti-commonsense > Commonsense",
            "Parametric memory should provide more compensation for commonsense scenarios"
        ],
        number_formats={
            'Delta2_avg': '{:+.3f}',
            # individual deltas formatted as {:+.3f}
        },
        column_order=[
            'Scenario', 'Model', 'n_questions', 'Delta2_avg'
        ]
    )
    
    return TablePackage(data=table_data, metadata=metadata)


# ============================================================================
# TABLE 4: RUNG-LEVEL ANALYSIS (H6)
# ============================================================================

def compute_table4_rung_analysis(data, model_name):
    """
    Compute rung-level analysis for H6.
    
    Tests: How effects vary by causal reasoning level (Pearl's hierarchy).
    Expected: Stronger effects at lower rungs (simpler reasoning).
    
    Returns:
        TablePackage with data DataFrame and formatting metadata
    """
    print("\n  Computing rung-level analysis...")
    
    rows = []
    for rung in RUNGS:
        rung_name = RUNG_NAMES[rung]
        print(f"    Rung {rung} ({rung_name})...")
        
        # filter to this rung
        rung_data = {k: v for k, v in data.items() if v['rung'] == rung}
        
        if not rung_data:
            continue
        
        # accuracies
        acc_ng, ci_ng, _, n_ng = compute_accuracy_with_ci(rung_data, condition='no_graph')
        acc_corr, ci_corr, _, n_corr = compute_accuracy_with_ci(
            rung_data, condition='original_graph')
        
        # deltas
        delta1, ci_d1, _ = bootstrap_delta(
            rung_data, 'original_graph', 'no_graph', n_bootstrap=5000)
        
        # average Δ₂ across corruptions with proper bootstrap CI
        delta2_avg, ci_d2_avg, _ = bootstrap_avg_delta2(
            rung_data, n_bootstrap=5000, verbose=False)
        
        rows.append({
            'Rung': rung,
            'Rung_Name': rung_name,
            'Model': model_name,
            'Acc_NoGraph': acc_ng,
            'Acc_NoGraph_CI_low': ci_ng[0],
            'Acc_NoGraph_CI_high': ci_ng[1],
            'n_NoGraph': n_ng,
            'Acc_Correct': acc_corr,
            'Acc_Correct_CI_low': ci_corr[0],
            'Acc_Correct_CI_high': ci_corr[1],
            'n_Correct': n_corr,
            'Delta1': delta1,
            'Delta1_CI_low': ci_d1[0] if ci_d1[0] is not None else None,
            'Delta1_CI_high': ci_d1[1] if ci_d1[1] is not None else None,
            'Delta2_avg': delta2_avg,
            'Delta2_avg_CI_low': ci_d2_avg[0] if ci_d2_avg and ci_d2_avg[0] is not None else None,
            'Delta2_avg_CI_high': ci_d2_avg[1] if ci_d2_avg and ci_d2_avg[1] is not None else None,
        })
    
    table_data = pd.DataFrame(rows)
    
    metadata = TableMetadata(
        title="Rung-Level Analysis (H6)",
        hypothesis="H6: Causal Reasoning Level Effects",
        footnotes=[
            "Rung 1 (Association): P(Y|X) - observational queries",
            "Rung 2 (Intervention): P(Y|do(X)) - action-based queries",
            "Rung 3 (Counterfactual): P(Y_x|X',Y') - what-if queries",
            "Expected: Stronger graph effects at lower rungs (simpler reasoning)"
        ],
        number_formats={
            'Acc_NoGraph': '{:.3f}',
            'Acc_Correct': '{:.3f}',
            'Delta1': '{:+.3f}',
            'Delta2_avg': '{:+.3f}',
        },
        column_order=[
            'Rung', 'Rung_Name', 'Acc_NoGraph', 'n_NoGraph',
            'Acc_Correct', 'n_Correct', 'Delta1', 'Delta2_avg'
        ],
        index_name='Rung'
    )
    
    return TablePackage(data=table_data, metadata=metadata)


# ============================================================================
# TABLE 5: ERROR PATTERN ANALYSIS (H4, H8)
# ============================================================================

def compute_table5_error_patterns(data, model_name):
    """
    Compute error pattern analysis for H4 and H8.
    
    Tests: Error types by condition to identify systematic patterns.
    Different corruptions should produce characteristic error patterns.
    
    Returns:
        TablePackage with data DataFrame and formatting metadata
    """
    print("\n  Computing error patterns...")
    
    # SOURCE: gen_stats.py error pattern computation
    # logic kept identical to source
    
    rows = []
    for condition in CONDITIONS:
        cond_data = {k: v for k, v in data.items() if v['condition'] == condition}
        
        if not cond_data:
            continue
        
        # count errors by type
        errors = {k: v for k, v in cond_data.items() if not v['is_correct']}
        
        # SOURCE: gen_stats.py error counting logic (identical)
        yes_to_no = sum(1 for v in errors.values() 
                       if v['ground_truth'] == 'yes' and v['response'] == 'no')
        no_to_yes = sum(1 for v in errors.values() 
                       if v['ground_truth'] == 'no' and v['response'] == 'yes')
        
        total_errors = len(errors)
        total = len(cond_data)
        error_rate = (total_errors / total) if total > 0 else 0
        
        rows.append({
            'Condition': condition,
            'Model': model_name,
            'Total': total,
            'Errors': total_errors,
            'Error_Rate': error_rate,
            'Yes_to_No': yes_to_no,
            'No_to_Yes': no_to_yes,
            'Ratio_YN_NY': yes_to_no / no_to_yes if no_to_yes > 0 else np.inf
        })
    
    table_data = pd.DataFrame(rows)
    
    metadata = TableMetadata(
        title="Error Pattern Analysis (H4, H8)",
        hypothesis="H4: Corruption-Type Specificity, H8: Error Consistency",
        footnotes=[
            "Yes→No: Model incorrectly predicted No when answer was Yes (false negative)",
            "No→Yes: Model incorrectly predicted Yes when answer was No (false positive)",
            "Ratio: (Yes→No) / (No→Yes) indicates bias direction",
            "Look for systematic patterns (e.g., colliders causing specific error types)"
        ],
        number_formats={
            'Error_Rate': '{:.1%}',
            'Ratio_YN_NY': '{:.2f}',
        },
        column_order=[
            'Condition', 'Total', 'Errors', 'Error_Rate',
            'Yes_to_No', 'No_to_Yes', 'Ratio_YN_NY'
        ],
        index_name='Condition'
    )
    
    return TablePackage(data=table_data, metadata=metadata)


# ============================================================================
# TABLE 6: QUESTION-LEVEL PATTERNS
# ============================================================================

def compute_table6_question_patterns(data, model_name):
    """
    Compute question-level correctness patterns.
    
    Shows distribution of correctness patterns across conditions.
    Pattern format: [no_graph][original][corrupted] where 1=correct, 0=wrong
    
    Key patterns for H1:
        - 111 (all correct): Easy question
        - 000 (all wrong): Hard question
        - 010 (only original correct): Graph helps
        - 110 (no_graph and original correct, corrupted wrong): Corruption hurts
        
    Returns:
        Tuple of (pattern_table, summary_table) as TablePackages
    """
    print("\n  Computing question-level patterns...")
    
    # SOURCE: gen_patterns.py analyze() function
    # logic kept identical to source
    
    # group by question
    by_question = defaultdict(lambda: {
        "no_graph": None, 
        "original_graph": None, 
        "corrupted": []
    })
    
    for uuid, v in data.items():
        qid = v['question_id']
        condition = v['condition']
        
        if condition == 'no_graph':
            by_question[qid]['no_graph'] = v['is_correct']
        elif condition == 'original_graph':
            by_question[qid]['original_graph'] = v['is_correct']
        elif condition in CORRUPTION_TYPES:
            by_question[qid]['corrupted'].append(v['is_correct'])
    
    # compute patterns
    # SOURCE: gen_patterns.py pattern computation (identical)
    pattern_counts = defaultdict(int)
    
    for qid, answers in by_question.items():
        ng = answers['no_graph']
        og = answers['original_graph']
        cg_list = answers['corrupted']
        
        # skip if missing key conditions
        if ng is None or og is None or not cg_list:
            continue
        
        # use strict criterion: all corruptions must be correct for cg=1
        cg = 1 if all(cg_list) else 0
        
        pattern = f"{int(ng)}{int(og)}{int(cg)}"
        pattern_counts[pattern] += 1
    
    total = sum(pattern_counts.values())
    
    # build pattern distribution table
    pattern_names = {
        "000": "All Wrong (Hard Question)",
        "001": "Only Corrupted Correct (Anomalous)",
        "010": "Only Original Correct (Graph Helps)",
        "011": "Original & Corrupted Correct",
        "100": "Only No-Graph Correct (Graph Hurts)",
        "101": "No-Graph & Corrupted Correct",
        "110": "No-Graph & Original Correct (Corruption Hurts)",
        "111": "All Correct (Easy Question)"
    }
    
    pattern_rows = []
    for pattern in sorted(pattern_counts.keys()):
        count = pattern_counts[pattern]
        pct = (count / total) if total > 0 else 0
        
        pattern_rows.append({
            'Pattern': pattern,
            'Interpretation': pattern_names.get(pattern, "Unknown"),
            'Model': model_name,
            'Count': count,
            'Percentage': pct
        })
    
    pattern_table = pd.DataFrame(pattern_rows)
    
    pattern_metadata = TableMetadata(
        title="Question-Level Correctness Patterns",
        hypothesis="Supporting analysis for H1",
        footnotes=[
            "Pattern format: [no_graph][original][corrupted] where 1=correct, 0=wrong",
            "Strict criterion: corrupted=1 only if ALL corruption types answered correctly"
        ],
        number_formats={
            'Percentage': '{:.1%}',
        },
        column_order=['Pattern', 'Interpretation', 'Count', 'Percentage'],
        index_name='Pattern'
    )
    
    # build summary metrics table
    # SOURCE: gen_patterns.py sensitivity metrics (identical)
    beneficial = pattern_counts.get("010", 0) + pattern_counts.get("011", 0)
    harmful = pattern_counts.get("100", 0) + pattern_counts.get("101", 0)
    corruption_sensitive = pattern_counts.get("110", 0)
    
    summary_rows = [{
        'Metric': 'Graph Beneficial (010, 011)',
        'Model': model_name,
        'Count': beneficial,
        'Percentage': beneficial / total if total > 0 else 0,
        'Description': 'Questions where graph helps'
    }, {
        'Metric': 'Graph Harmful (100, 101)',
        'Model': model_name,
        'Count': harmful,
        'Percentage': harmful / total if total > 0 else 0,
        'Description': 'Questions where graph hurts'
    }, {
        'Metric': 'Corruption Sensitive (110)',
        'Model': model_name,
        'Count': corruption_sensitive,
        'Percentage': corruption_sensitive / total if total > 0 else 0,
        'Description': 'Correct with no_graph and original, wrong with corrupted'
    }, {
        'Metric': 'Net Graph Effect',
        'Model': model_name,
        'Count': beneficial - harmful,
        'Percentage': (beneficial - harmful) / total if total > 0 else 0,
        'Description': 'Beneficial - Harmful'
    }]
    
    summary_table = pd.DataFrame(summary_rows)
    
    summary_metadata = TableMetadata(
        title="Pattern Summary Metrics",
        hypothesis="H1 supporting metrics",
        footnotes=[
            "Net positive effect indicates graphs are beneficial overall",
            "Net negative effect indicates graphs are harmful overall"
        ],
        number_formats={
            'Percentage': '{:.1%}',
        },
        column_order=['Metric', 'Count', 'Percentage', 'Description']
    )
    
    return (
        TablePackage(data=pattern_table, metadata=pattern_metadata),
        TablePackage(data=summary_table, metadata=summary_metadata)
    )


# ============================================================================
# TABLE 7: STATISTICAL TESTS (Supplementary)
# ============================================================================

def compute_table7_statistical_tests(data, model_name, n_bootstrap=5000):
    """
    Compute supplementary statistical tests for paper.
    
    Tests included:
    1. Rung-level ANOVA for Δ₁ (tests if graph benefit varies by rung)
    2. Rung-level ANOVA for Δ₂ (tests if corruption sensitivity varies by rung)
    3. Corruption type pairwise comparisons (tests ranking significance)
    4. McNemar's test for paired outcomes
    5. Effect sizes (Cohen's h)
    
    This function leverages the same bootstrap infrastructure as other tables,
    avoiding duplicate computation.
    
    Returns:
        TablePackage with statistical test results
    """
    print("\n  Computing statistical tests...")
    
    results = {}
    
    # =========================================================================
    # TEST 1: RUNG-LEVEL ANOVA FOR Δ₁
    # =========================================================================
    print("    [1/5] Rung-level ANOVA for Δ₁...")
    
    rung_anova_results = _test_rung_anova(data, n_bootstrap)
    results['rung_anova'] = rung_anova_results
    
    # =========================================================================
    # TEST 1b: RUNG-LEVEL ANOVA FOR Δ₂
    # =========================================================================
    print("    [2/5] Rung-level ANOVA for Δ₂...")
    
    rung_anova_delta2_results = _test_rung_anova_delta2(data, n_bootstrap)
    results['rung_anova_delta2'] = rung_anova_delta2_results
    
    # =========================================================================
    # TEST 2: CORRUPTION TYPE PAIRWISE COMPARISONS
    # =========================================================================
    print("    [3/5] Corruption pairwise comparisons...")
    
    corruption_pairwise_results = _test_corruption_pairwise(data, n_bootstrap)
    results['corruption_pairwise'] = corruption_pairwise_results
    
    # =========================================================================
    # TEST 3: MCNEMAR'S TEST
    # =========================================================================
    print("    [4/5] McNemar's tests...")
    
    mcnemar_results = _test_mcnemar(data)
    results['mcnemar'] = mcnemar_results
    
    # =========================================================================
    # TEST 4: EFFECT SIZES
    # =========================================================================
    print("    [5/5] Effect sizes...")
    
    effect_size_results = _test_effect_sizes(data)
    results['effect_sizes'] = effect_size_results
    
    # =========================================================================
    # CREATE OUTPUT TABLE
    # =========================================================================
    
    # Format results as human-readable text table
    table_rows = []
    
    # Rung ANOVA Δ₁ section
    table_rows.append({
        'Test': 'Rung-Level ANOVA (Δ₁)',
        'Statistic': f"F = {rung_anova_results['f_stat']:.2f}",
        'p_value': f"{rung_anova_results['p_value']:.4f}",
        'Interpretation': _interpret_p(rung_anova_results['p_value'])
    })
    
    # Key rung pairwise for Δ₁
    for comparison, vals in rung_anova_results['pairwise'].items():
        table_rows.append({
            'Test': f"  {comparison}",
            'Statistic': f"Δ = {vals['difference']:+.3f}",
            'p_value': f"{vals['p_value']:.4f}",
            'Interpretation': _interpret_p(vals['p_value'])
        })
    
    # Rung ANOVA Δ₂ section
    table_rows.append({
        'Test': 'Rung-Level ANOVA (Δ₂)',
        'Statistic': f"F = {rung_anova_delta2_results['f_stat']:.2f}",
        'p_value': f"{rung_anova_delta2_results['p_value']:.4f}",
        'Interpretation': _interpret_p(rung_anova_delta2_results['p_value'])
    })
    
    # Key rung pairwise for Δ₂
    for comparison, vals in rung_anova_delta2_results['pairwise'].items():
        table_rows.append({
            'Test': f"  {comparison}",
            'Statistic': f"Δ = {vals['difference']:+.3f}",
            'p_value': f"{vals['p_value']:.4f}",
            'Interpretation': _interpret_p(vals['p_value'])
        })
    
    # corruption pairwise
    table_rows.append({
        'Test': 'Corruption Pairwise',
        'Statistic': '',
        'p_value': '',
        'Interpretation': ''
    })
    
    for comparison, vals in corruption_pairwise_results['pairwise'].items():
        table_rows.append({
            'Test': f"  {comparison}",
            'Statistic': f"Δ = {vals['difference']:+.3f}",
            'p_value': f"{vals['p_value']:.4f}",
            'Interpretation': _interpret_p(vals['p_value'], bonf=vals.get('p_bonferroni', False))
        })
    
    # McNemar's test
    table_rows.append({
        'Test': "McNemar's Test",
        'Statistic': '',
        'p_value': '',
        'Interpretation': ''
    })
    
    if mcnemar_results.get('original_vs_no_graph'):
        data = mcnemar_results['original_vs_no_graph']
        table_rows.append({
            'Test': "  original vs. no_graph",
            'Statistic': f"χ² = {data['mcnemar_stat']:.2f}",
            'p_value': f"{data['p_value']:.4f}",
            'Interpretation': _interpret_p(data['p_value'])
        })
    
    if mcnemar_results.get('original_vs_corrupted'):
        data = mcnemar_results['original_vs_corrupted']
        table_rows.append({
            'Test': "  original vs. corrupted",
            'Statistic': f"χ² = {data['mcnemar_stat']:.2f}",
            'p_value': f"{data['p_value']:.4f}",
            'Interpretation': _interpret_p(data['p_value'])
        })
    
    if mcnemar_results.get('no_graph_vs_corrupted'):
        data = mcnemar_results['no_graph_vs_corrupted']
        table_rows.append({
            'Test': "  no_graph vs. corrupted",
            'Statistic': f"χ² = {data['mcnemar_stat']:.2f}",
            'p_value': f"{data['p_value']:.4f}",
            'Interpretation': _interpret_p(data['p_value'])
        })
    
    # Effect sizes - show all
    table_rows.append({
        'Test': 'Effect Sizes',
        'Statistic': '',
        'p_value': '',
        'Interpretation': ''
    })
    
    if 'delta1' in effect_size_results:
        table_rows.append({
            'Test': "  Δ₁ (original vs. no_graph)",
            'Statistic': f"h = {effect_size_results['delta1']['cohens_h']:.3f}",
            'p_value': '-',
            'Interpretation': effect_size_results['delta1']['interpretation']
        })
    
    if 'delta2_max' in effect_size_results:
        table_rows.append({
            'Test': "  Δ₂ (original vs. corrupted)",
            'Statistic': f"h = {effect_size_results['delta2_max']['cohens_h']:.3f}",
            'p_value': '-',
            'Interpretation': effect_size_results['delta2_max']['interpretation']
        })
    
    if 'delta3_avg' in effect_size_results:
        table_rows.append({
            'Test': "  Δ₃ (no_graph vs. corrupted)",
            'Statistic': f"h = {effect_size_results['delta3_avg']['cohens_h']:.3f}",
            'p_value': '-',
            'Interpretation': effect_size_results['delta3_avg']['interpretation']
        })
    
    table_data = pd.DataFrame(table_rows)
    
    # detailed results as pickle
    metadata = TableMetadata(
        title="Statistical Tests Summary",
        hypothesis="Supplementary statistical tests",
        footnotes=[
            "Test 1: Rung-level ANOVA tests if Δ₁ (graph benefit) varies significantly by rung",
            "Test 2: Rung-level ANOVA tests if Δ₂ (corruption harm) varies significantly by rung",
            "Test 3: All pairwise corruption type comparisons with Bonferroni correction",
            "Test 4: McNemar's test for paired binary outcomes (three comparisons)",
            "Test 5: Cohen's h effect sizes for Δ₁ (original vs. no_graph), Δ₂ (original vs. corrupted), and Δ₃ (no_graph vs. corrupted)",
            f"Bootstrap iterations: {n_bootstrap}",
        ],
        column_order=['Test', 'Statistic', 'p_value', 'Interpretation']
    )
    
    # full results in metadata for pickle
    metadata.full_results = results
    
    return TablePackage(data=table_data, metadata=metadata)


def _test_rung_anova(data, n_bootstrap):
    """
    Test if Δ₁ varies significantly across rungs using bootstrap ANOVA.
    Uses same bootstrap pattern as rest of project: resample once, compute all stats.
    """
    # compute Δ₁ for each rung (reusing existing infrastructure)
    rung_deltas = {}
    for rung in RUNGS:
        rung_data = {k: v for k, v in data.items() if v['rung'] == rung}
        if rung_data:
            delta1, ci, _ = bootstrap_delta(
                rung_data, 'original_graph', 'no_graph', n_bootstrap=n_bootstrap)
            rung_deltas[rung] = {'delta1': delta1, 'ci': ci}
    
    # compute F-statistic on question-level deltas
    f_stat = _compute_anova_f_stat(data, RUNGS)
    
    # bootstrap p-value (permutation test) - same pattern as bootstrap_delta
    by_question = build_question_index(data)
    questions = list(by_question.keys())
    
    null_f_stats = []
    for i in range(min(n_bootstrap, 1000)):  # reduce for ANOVA (still robust)
        if (i + 1) % 200 == 0:
            print(f"      ANOVA bootstrap {i+1}/1000", end='\r')
        resampled = resample_by_questions(by_question, questions)
        f_null = _compute_anova_f_stat(resampled, RUNGS)
        if f_null is not None:
            null_f_stats.append(f_null)
    
    print(" " * 60, end='\r') 
    p_value = np.mean([f >= f_stat for f in null_f_stats]) if null_f_stats else None
    
    # Pairwise comparisons using bootstrap distribution of differences
    # FIX: resample once, compute both deltas from same resample
    pairwise = {}
    for r1, r2 in [(2, 1), (2, 3)]:  # focus on Rung 2 vs others
        # indices for each rung
        r1_by_q = build_question_index({k: v for k, v in data.items() if v['rung'] == r1})
        r2_by_q = build_question_index({k: v for k, v in data.items() if v['rung'] == r2})
        r1_questions = list(r1_by_q.keys())
        r2_questions = list(r2_by_q.keys())
        
        if not r1_questions or not r2_questions:
            continue
        
        d1_r1 = rung_deltas[r1]['delta1']
        d1_r2 = rung_deltas[r2]['delta1']
        diff = d1_r1 - d1_r2
        
        # bootstrap distribution of difference: resample BOTH independently
        diffs = []
        for _ in range(min(n_bootstrap, 5000)):
            # resample each rung independently
            r1_boot = resample_by_questions(r1_by_q, r1_questions)
            r2_boot = resample_by_questions(r2_by_q, r2_questions)
            
            # compute deltas from resampled data
            acc_r1_cg, _, _, _ = compute_accuracy_with_ci(r1_boot, condition='original_graph')
            acc_r1_ng, _, _, _ = compute_accuracy_with_ci(r1_boot, condition='no_graph')
            acc_r2_cg, _, _, _ = compute_accuracy_with_ci(r2_boot, condition='original_graph')
            acc_r2_ng, _, _, _ = compute_accuracy_with_ci(r2_boot, condition='no_graph')
            
            if all(x is not None for x in [acc_r1_cg, acc_r1_ng, acc_r2_cg, acc_r2_ng]):
                d1_b = acc_r1_cg - acc_r1_ng
                d2_b = acc_r2_cg - acc_r2_ng
                diffs.append(d1_b - d2_b)
        
        if diffs:
            ci_diff = (np.percentile(diffs, 2.5), np.percentile(diffs, 97.5))
            p_pair = np.mean([d <= 0 for d in diffs]) if diff > 0 else np.mean([d >= 0 for d in diffs])
            p_pair = min(p_pair * 2, 1.0)  # Two-tailed
            
            pairwise[f"Rung_{r1}_vs_{r2}"] = {
                'difference': diff,
                'ci': ci_diff,
                'p_value': p_pair
            }
    
    return {
        'rung_deltas': rung_deltas,
        'f_stat': f_stat,
        'p_value': p_value,
        'pairwise': pairwise
    }


def _compute_anova_f_stat(data, rungs):
    """Compute F-statistic for Δ₁ across rungs."""
    rung_delta1s = defaultdict(list)
    
    for rung in rungs:
        rung_data = {k: v for k, v in data.items() if v['rung'] == rung}
        
        # group by question
        by_q = defaultdict(lambda: {'ng': None, 'cg': None})
        for uuid, v in rung_data.items():
            qid = v['question_id']
            if v['condition'] == 'no_graph':
                by_q[qid]['ng'] = v['is_correct']
            elif v['condition'] == 'original_graph':
                by_q[qid]['cg'] = v['is_correct']
        
        # compute question-level Δ₁
        for qid, answers in by_q.items():
            if answers['ng'] is not None and answers['cg'] is not None:
                delta1 = int(answers['cg']) - int(answers['ng'])
                rung_delta1s[rung].append(delta1)
    
    # compute F-statistic
    if any(len(vals) < 2 for vals in rung_delta1s.values()):
        return None
    
    try:
        values = [rung_delta1s[r] for r in rungs if r in rung_delta1s]
        f_stat, _ = stats.f_oneway(*values)
        return f_stat
    except:
        return None


def _test_rung_anova_delta2(data, n_bootstrap):
    """
    Test if Δ₂ varies significantly across rungs using bootstrap ANOVA.
    Parallel to _test_rung_anova but for corruption harm instead of graph benefit.
    Uses same bootstrap pattern: resample once, compute all stats.
    """
    # compute Δ₂ (average across corruption types) for each rung
    rung_deltas = {}
    for rung in RUNGS:
        rung_data = {k: v for k, v in data.items() if v['rung'] == rung}
        if rung_data:
            # compute average Δ₂ for this rung
            delta2_avg, ci, _ = bootstrap_avg_delta2(rung_data, n_bootstrap=n_bootstrap, verbose=False)
            rung_deltas[rung] = {'delta2_avg': delta2_avg, 'ci': ci}
    
    # compute F-statistic on question-level Δ₂s
    f_stat = _compute_anova_f_stat_delta2(data, RUNGS)
    
    # bootstrap p-value (permutation test) - (VALIDATION NOTE: same pattern as Δ₁)
    by_question = build_question_index(data)
    questions = list(by_question.keys())
    
    null_f_stats = []
    for i in range(min(n_bootstrap, 1000)):  # reduce for ANOVA (still robust)
        if (i + 1) % 200 == 0:
            print(f"      ANOVA Δ₂ bootstrap {i+1}/1000", end='\r')
        resampled = resample_by_questions(by_question, questions)
        f_null = _compute_anova_f_stat_delta2(resampled, RUNGS)
        if f_null is not None:
            null_f_stats.append(f_null)
    
    print(" " * 60, end='\r') 
    p_value = np.mean([f >= f_stat for f in null_f_stats]) if null_f_stats else None
    
    # Pairwise comparisons using bootstrap distribution of differences
    pairwise = {}
    for r1, r2 in [(2, 1), (2, 3)]:  # focus on Rung 2 vs others
        # indices for each rung
        r1_by_q = build_question_index({k: v for k, v in data.items() if v['rung'] == r1})
        r2_by_q = build_question_index({k: v for k, v in data.items() if v['rung'] == r2})
        r1_questions = list(r1_by_q.keys())
        r2_questions = list(r2_by_q.keys())
        
        if not r1_questions or not r2_questions:
            continue
        
        d2_r1 = rung_deltas[r1]['delta2_avg']
        d2_r2 = rung_deltas[r2]['delta2_avg']
        diff = d2_r1 - d2_r2
        
        # bootstrap distribution of difference: resample BOTH independently
        diffs = []
        for _ in range(min(n_bootstrap, 5000)):
            # resample each rung independently
            r1_boot = resample_by_questions(r1_by_q, r1_questions)
            r2_boot = resample_by_questions(r2_by_q, r2_questions)
            
            # compute average delta2 for each rung from resampled data
            d2_vals_r1 = []
            d2_vals_r2 = []
            
            for corr_type in CORRUPTION_TYPES:
                # Rung 1
                acc_r1_cg, _, _, _ = compute_accuracy_with_ci(r1_boot, condition='original_graph')
                acc_r1_corr, _, _, _ = compute_accuracy_with_ci(r1_boot, condition=corr_type)
                if acc_r1_cg is not None and acc_r1_corr is not None:
                    d2_vals_r1.append(acc_r1_cg - acc_r1_corr)
                
                # Rung 2
                acc_r2_cg, _, _, _ = compute_accuracy_with_ci(r2_boot, condition='original_graph')
                acc_r2_corr, _, _, _ = compute_accuracy_with_ci(r2_boot, condition=corr_type)
                if acc_r2_cg is not None and acc_r2_corr is not None:
                    d2_vals_r2.append(acc_r2_cg - acc_r2_corr)
            
            if d2_vals_r1 and d2_vals_r2:
                d2_avg_r1 = np.mean(d2_vals_r1)
                d2_avg_r2 = np.mean(d2_vals_r2)
                diffs.append(d2_avg_r1 - d2_avg_r2)
        
        if diffs:
            ci_diff = (np.percentile(diffs, 2.5), np.percentile(diffs, 97.5))
            p_pair = np.mean([d <= 0 for d in diffs]) if diff > 0 else np.mean([d >= 0 for d in diffs])
            p_pair = min(p_pair * 2, 1.0)  # Two-tailed
            
            pairwise[f"Rung_{r1}_vs_{r2}"] = {
                'difference': diff,
                'ci': ci_diff,
                'p_value': p_pair
            }
    
    return {
        'rung_deltas': rung_deltas,
        'f_stat': f_stat,
        'p_value': p_value,
        'pairwise': pairwise
    }


def _compute_anova_f_stat_delta2(data, rungs):
    """Compute F-statistic for average Δ₂ across rungs."""
    rung_delta2s = defaultdict(list)
    
    for rung in rungs:
        rung_data = {k: v for k, v in data.items() if v['rung'] == rung}
        
        # group by question
        by_q = defaultdict(lambda: {'cg': None, 'corrupted': []})
        for uuid, v in rung_data.items():
            qid = v['question_id']
            if v['condition'] == 'original_graph':
                by_q[qid]['cg'] = v['is_correct']
            elif v['condition'] in CORRUPTION_TYPES:
                by_q[qid]['corrupted'].append(v['is_correct'])
        
        # compute question-level average Δ₂
        for qid, answers in by_q.items():
            if answers['cg'] is not None and answers['corrupted']:
                # average Δ₂ across all corruption types for this question
                delta2s = []
                for corr_correct in answers['corrupted']:
                    delta2 = int(answers['cg']) - int(corr_correct)
                    delta2s.append(delta2)
                rung_delta2s[rung].append(np.mean(delta2s))
    
    # compute F-statistic
    if any(len(vals) < 2 for vals in rung_delta2s.values()):
        return None
    
    try:
        values = [rung_delta2s[r] for r in rungs if r in rung_delta2s]
        f_stat, _ = stats.f_oneway(*values)
        return f_stat
    except:
        return None


def _test_corruption_pairwise(data, n_bootstrap):
    """
    Test pairwise differences in corruption sensitivity.
    Uses same bootstrap pattern as rest of project: resample once, compute all stats.
    FIX: Do NOT call bootstrap_delta in a loop
    """
    # compute Δ₂ for each corruption (reusing infrastructure)
    corruption_deltas = {}
    for corr_type in CORRUPTION_TYPES:
        delta2, ci, _ = bootstrap_delta(
            data, 'original_graph', corr_type, n_bootstrap=n_bootstrap)
        corruption_deltas[corr_type] = {'delta2': delta2, 'ci': ci}
    
    # key pairwise comparisons (not all pairs - too slow) #TODO
    key_pairs = [
        ('reverse_random_edge', 'add_confounder'),  # Most important
        ('reverse_random_edge', 'add_collider'),
        ('add_confounder', 'add_collider'),
        ('add_collider', 'add_mediator')
    ]
    
    pairwise = {}
    n_comparisons = len(key_pairs)
    bonferroni_alpha = 0.05 / n_comparisons
    
    # build single index for all data (more efficient than per-corruption)
    by_question = build_question_index(data)
    questions = list(by_question.keys())
    
    for pair_idx, (c1, c2) in enumerate(key_pairs):
        print(f"      Pair {pair_idx+1}/{n_comparisons}: {c1} vs {c2}", end='\r')
        
        diff = corruption_deltas[c1]['delta2'] - corruption_deltas[c2]['delta2']
        
        # bootstrap distribution of difference: same pattern as bootstrap_delta
        # FIX: resample data ONCE, compute BOTH deltas from that resample
        diffs = []
        for i in range(min(n_bootstrap, 5000)):
            if (i + 1) % 500 == 0:
                print(f"      Pair {pair_idx+1}/{n_comparisons}: bootstrap {i+1}/{min(n_bootstrap, 5000)}", end='\r')
            
            # resamp once for this iteration
            resampled = resample_by_questions(by_question, questions)
            
            # comp BOTH deltas from the SAME resample (paired bootstrap)
            acc_cg, _, _, _ = compute_accuracy_with_ci(resampled, condition='original_graph')
            acc_c1, _, _, _ = compute_accuracy_with_ci(resampled, condition=c1)
            acc_c2, _, _, _ = compute_accuracy_with_ci(resampled, condition=c2)
            
            if all(x is not None for x in [acc_cg, acc_c1, acc_c2]):
                d1 = acc_cg - acc_c1
                d2 = acc_cg - acc_c2
                diffs.append(d1 - d2)
        
        print(" " * 80, end='\r') 
        
        if diffs:
            ci_diff = (np.percentile(diffs, 2.5), np.percentile(diffs, 97.5))
            p_value = np.mean([d <= 0 for d in diffs]) if diff > 0 else np.mean([d >= 0 for d in diffs])
            p_value = min(p_value * 2, 1.0)  # Two-tailed
            
            c1_short = c1.replace('_', ' ').replace('add ', '').replace('reverse random ', '')
            c2_short = c2.replace('_', ' ').replace('add ', '').replace('reverse random ', '')
            
            pairwise[f"{c1_short}_vs_{c2_short}"] = {
                'difference': diff,
                'ci': ci_diff,
                'p_value': p_value,
                'p_bonferroni': p_value < bonferroni_alpha
            }
    
    print(" " * 80, end='\r')
    
    return {
        'corruption_deltas': corruption_deltas,
        'pairwise': pairwise,
        'bonferroni_alpha': bonferroni_alpha
    }


def _test_mcnemar(data):
    """McNemar's test for paired binary outcomes."""
    # group by question
    by_question = defaultdict(lambda: {
        'no_graph': None,
        'original_graph': None,
        'corrupted': {}
    })
    
    for uuid, v in data.items():
        qid = v['question_id']
        if v['condition'] == 'no_graph':
            by_question[qid]['no_graph'] = v['is_correct']
        elif v['condition'] == 'original_graph':
            by_question[qid]['original_graph'] = v['is_correct']
        elif v['condition'] in CORRUPTION_TYPES:
            by_question[qid]['corrupted'][v['condition']] = v['is_correct']
    
    results = {}
    
    # Test 1: Original (correct) graph vs. No graph
    contingency = {'both_correct': 0, 'only_ng': 0, 'only_cg': 0, 'both_wrong': 0}
    
    for qid, answers in by_question.items():
        ng = answers['no_graph']
        cg = answers['original_graph']
        
        if ng is not None and cg is not None:
            if cg and ng:
                contingency['both_correct'] += 1
            elif cg and not ng:
                contingency['only_cg'] += 1
            elif not cg and ng:
                contingency['only_ng'] += 1
            else:
                contingency['both_wrong'] += 1
    
    b = contingency['only_ng']
    c = contingency['only_cg']
    
    if b + c > 0:
        mcnemar_stat = ((abs(b - c) - 1) ** 2) / (b + c)
        p_value = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
        
        results['original_vs_no_graph'] = {
            'contingency': contingency,
            'mcnemar_stat': mcnemar_stat,
            'p_value': p_value
        }
    
    # Test 2: Original (correct) graph vs. Corrupted (any corruption, averaged)
    contingency = {'both_correct': 0, 'only_corr': 0, 'only_cg': 0, 'both_wrong': 0}
    
    for qid, answers in by_question.items():
        cg = answers['original_graph']
        
        # avg across all corruptions for this question
        corr_answers = list(answers['corrupted'].values())
        if corr_answers:
            avg_corr = np.mean(corr_answers)
            corr_correct = avg_corr >= 0.5  # >=50% of corruptions correct
            
            if cg is not None:
                if cg and corr_correct:
                    contingency['both_correct'] += 1
                elif cg and not corr_correct:
                    contingency['only_cg'] += 1
                elif not cg and corr_correct:
                    contingency['only_corr'] += 1
                else:
                    contingency['both_wrong'] += 1
    
    b = contingency['only_corr']
    c = contingency['only_cg']
    
    if b + c > 0:
        mcnemar_stat = ((abs(b - c) - 1) ** 2) / (b + c)
        p_value = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
        
        results['original_vs_corrupted'] = {
            'contingency': contingency,
            'mcnemar_stat': mcnemar_stat,
            'p_value': p_value
        }
    
    # Test 3: No graph vs. Corrupted (any corruption, averaged)
    contingency = {'both_correct': 0, 'only_corr': 0, 'only_ng': 0, 'both_wrong': 0}
    
    for qid, answers in by_question.items():
        ng = answers['no_graph']
        
        # avg across all corruptions for this question
        corr_answers = list(answers['corrupted'].values())
        if corr_answers:
            avg_corr = np.mean(corr_answers)
            corr_correct = avg_corr >= 0.5  # >=50% of corruptions correct
            
            if ng is not None:
                if ng and corr_correct:
                    contingency['both_correct'] += 1
                elif ng and not corr_correct:
                    contingency['only_ng'] += 1
                elif not ng and corr_correct:
                    contingency['only_corr'] += 1
                else:
                    contingency['both_wrong'] += 1
    
    b = contingency['only_corr']
    c = contingency['only_ng']
    
    if b + c > 0:
        mcnemar_stat = ((abs(b - c) - 1) ** 2) / (b + c)
        p_value = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
        
        results['no_graph_vs_corrupted'] = {
            'contingency': contingency,
            'mcnemar_stat': mcnemar_stat,
            'p_value': p_value
        }
    
    return results


def _test_effect_sizes(data):
    """Compute Cohen's h effect sizes."""
    results = {}
    
    # Δ₁ effect size: original graph vs no graph
    acc_ng, _, _, _ = compute_accuracy_with_ci(data, condition='no_graph')
    acc_cg, _, _, _ = compute_accuracy_with_ci(data, condition='original_graph')
    
    if acc_ng is not None and acc_cg is not None:
        h = 2 * (np.arcsin(np.sqrt(acc_cg)) - np.arcsin(np.sqrt(acc_ng)))
        interpretation = 'large' if abs(h) >= 0.5 else 'medium' if abs(h) >= 0.2 else 'small'
        
        results['delta1'] = {
            'cohens_h': h,
            'interpretation': interpretation
        }
    
    # Δ₂ effect sizes: original graph vs corrupted (store max across corruption types)
    max_h = 0
    max_corr = None
    for corr_type in CORRUPTION_TYPES:
        acc_corr, _, _, _ = compute_accuracy_with_ci(data, condition=corr_type)
        
        if acc_cg is not None and acc_corr is not None:
            h = 2 * (np.arcsin(np.sqrt(acc_cg)) - np.arcsin(np.sqrt(acc_corr)))
            if abs(h) > abs(max_h):
                max_h = h
                max_corr = corr_type
    
    if max_corr:
        interpretation = 'large' if abs(max_h) >= 0.5 else 'medium' if abs(max_h) >= 0.2 else 'small'
        results['delta2_max'] = {
            'corruption': max_corr,
            'cohens_h': max_h,
            'interpretation': interpretation
        }
    
    # Δ₃ effect size: no graph vs corrupted (avg effect across all corruption types)
    if acc_ng is not None:
        h_values = []
        for corr_type in CORRUPTION_TYPES:
            acc_corr, _, _, _ = compute_accuracy_with_ci(data, condition=corr_type)
            if acc_corr is not None:
                h = 2 * (np.arcsin(np.sqrt(acc_ng)) - np.arcsin(np.sqrt(acc_corr)))
                h_values.append(h)
        
        if h_values:
            avg_h = np.mean(h_values)
            interpretation = 'large' if abs(avg_h) >= 0.5 else 'medium' if abs(avg_h) >= 0.2 else 'small'
            
            results['delta3_avg'] = {
                'cohens_h': avg_h,
                'interpretation': interpretation
            }
    
    return results


def _interpret_p(p, bonf=False):
    """Interpret p-value."""
    if p < 0.001:
        sig = '***'
    elif p < 0.01:
        sig = '**'
    elif p < 0.05:
        sig = '*'
    else:
        sig = 'ns'
    
    if bonf and sig != 'ns':
        sig += ' (Bonf)'
    
    return sig



# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate hypothesis testing tables',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # generate plaintext tables for GPT-5-mini
    python generate_hypothesis_tables_v2_final.py \\
        --metadata ./data/corruption/dataset.jsonl \\
        --csv ./data/model/gpt5_responses.csv \\
        --model-name "GPT-5-mini" \\
        --format txt

    # save raw data as pickle for future/custom formatting
    python generate_hypothesis_tables_v2_final.py \\
        --metadata ./data/corruption/dataset.jsonl \\
        --csv ./data/model/claude_responses.csv \\
        --model-name "Claude Sonnet 4" \\
        --save-raw
        """
    )
    parser.add_argument('--metadata', required=True, help='JSONL metadata file')
    parser.add_argument('--csv', required=True, help='Model responses CSV file')
    parser.add_argument('--model-name', required=True, 
                        help='Model name for tables (e.g., "GPT-5-mini")')
    parser.add_argument('--output-dir', default='./tables', 
                        help='Output directory for tables')
    parser.add_argument('--format', default='txt', 
                        choices=['txt', 'csv'],
                        help='Output format (default: txt = clean plaintext)')
    parser.add_argument('--save-raw', action='store_true',
                        help='Also save raw TablePackages as pickle')
    parser.add_argument('--n-bootstrap', type=int, default=10000,
                        help='Number of bootstrap iterations (default: 10000)')
    args = parser.parse_args()
    
    # create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("GENERATING HYPOTHESIS TESTING TABLES")
    print("=" * 70)
    print(f"\nModel: {args.model_name}")
    print(f"Metadata: {args.metadata}")
    print(f"Responses: {args.csv}")
    print(f"Output: {output_dir}")
    print(f"Format: {args.format}")
    
    # load data
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    data = load_and_structure_data(args.metadata, args.csv)
    
    # generate tables (compute data + metadata)
    print("\n" + "=" * 70)
    print("COMPUTING TABLES")
    print("=" * 70)
    
    table_packages = {}
    
    print("\n[1/6] Table 1: Overall Performance & Delta Metrics (H1)")
    table_packages['table1'] = compute_table1_overall_performance(data, args.model_name)
    
    print("\n[2/6] Table 2: Corruption-Specific Sensitivity (H2)")
    table_packages['table2'] = compute_table2_corruption_specific(data, args.model_name)
    
    print("\n[3/6] Table 3: Scenario × Corruption Interaction (H3)")
    table_packages['table3'] = compute_table3_scenario_interaction(data, args.model_name)
    
    print("\n[4/6] Table 4: Rung-Level Analysis (H6)")
    table_packages['table4'] = compute_table4_rung_analysis(data, args.model_name)
    
    print("\n[5/6] Table 5: Error Pattern Analysis (H4, H8)")
    table_packages['table5'] = compute_table5_error_patterns(data, args.model_name)
    
    print("\n[6/7] Table 6: Question-Level Patterns")
    table6_patterns, table6_summary = compute_table6_question_patterns(data, args.model_name)
    table_packages['table6_patterns'] = table6_patterns
    table_packages['table6_summary'] = table6_summary
    
    print("\n[7/7] Table 7: Statistical Tests")
    table_packages['table7'] = compute_table7_statistical_tests(
        data, args.model_name, n_bootstrap=args.n_bootstrap)
    
    # save raw table packages if requested
    if args.save_raw:
        print("\n" + "=" * 70)
        print("SAVING RAW TABLE PACKAGES")
        print("=" * 70)
        
        model_slug = args.model_name.lower().replace(' ', '_').replace('-', '_')
        raw_path = output_dir / f"{model_slug}_raw_tables.pkl"
        
        with open(raw_path, 'wb') as f:
            pickle.dump(table_packages, f)
        
        print(f"   Raw tables saved to: {raw_path}")
        print("  Load with: pickle.load(open('file.pkl', 'rb'))")
    
    # format and save tables
    print("\n" + "=" * 70)
    print("FORMATTING AND SAVING TABLES")
    print("=" * 70)
    
    model_slug = args.model_name.lower().replace(' ', '_').replace('-', '_')
    
    for name, table_pkg in table_packages.items():
        base_name = f"{model_slug}_{name}"
        
        # determine file extension
        if args.format == 'csv':
            ext = 'csv'
        else:  # txt
            ext = 'txt'
        
        output_file = output_dir / f"{base_name}.{ext}"
        
        # format table using helper function
        formatted = format_table_for_output(table_pkg, args.format)
        
        # write to file
        with open(output_file, 'w') as f:
            f.write(formatted)
        
        print(f"   {output_file.name}")
    
    # generate a summary README
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(f"# Hypothesis Testing Tables: {args.model_name}\n\n")
        f.write("## Table Descriptions\n\n")
        
        for name, table_pkg in table_packages.items():
            meta = table_pkg.metadata
            f.write(f"### {meta.title}\n")
            f.write(f"**Hypothesis:** {meta.hypothesis}\n\n")
            if meta.footnotes:
                f.write("**Notes:**\n")
                for note in meta.footnotes:
                    f.write(f"- {note}\n")
            f.write("\n")
        
        f.write("## Data Summary\n\n")
        f.write(f"- Total valid responses: {len(data)}\n")
        f.write(f"- Unique questions: {len(get_unique_questions(data))}\n")
        f.write(f"- Conditions tested: {len(CONDITIONS)}\n")
    
    print(f"   {readme_path.name}")
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nAll tables saved to: {output_dir}/")
    print(f"See {readme_path.name} for table descriptions")
    
    if args.save_raw:
        print(f"\nRaw table packages: {raw_path}")


if __name__ == '__main__':
    main()