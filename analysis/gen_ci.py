#!/usr/bin/env python3
"""
Confidence Intervals for Causal Reasoning Eval

Computes CIs for accuracy metrics and derived statistics.
Uses the simplest appropriate method for each metric type:
  - Binomial CI: for simple accuracy proportions (exact, fast)
  - Bootstrap: for deltas and pattern metrics (necessary for complex derived stats)

Usage:
    python analysis/gen_ci.py \
        --metadata data/corruption/corrupted_causal_graphs_dataset_f1-3.jsonl \
        --responses data/model/claude_responses_f1-3.csv \
        --model-name claude \
        --n-bootstrap 10000 \
        --output ./analysis/tables/_raw/claude_ci.txt
"""


import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from statsmodels.stats.proportion import proportion_confint
from analysis_common import (
    load_metadata_dict,
    load_responses_dict,
    filter_valid_data,
    print_data_summary,
    get_condition,
    get_unique_questions,
    CONDITIONS,
    RUNGS,
    SENSE_CATEGORIES,
    CORRUPTION_TYPES,
)



# =============================================================================
# DATA LOADING
# =============================================================================

def load_metadata(jsonl_path):
    return load_metadata_dict(jsonl_path)


def load_responses(csv_path):
    return load_responses_dict(csv_path)

# =============================================================================
# BINOMIAL CONFIDENCE INTERVALS (for simple accuracies)
# =============================================================================

def binomial_ci(n_correct, n_total, alpha=0.05):
    """
    Compute exact binomial CI using Wilson score interval.
    """
    if n_total == 0:
        return 0.0, (0.0, 0.0)
    
    accuracy = n_correct / n_total
    ci_lower, ci_upper = proportion_confint(
        count=n_correct,
        nobs=n_total,
        alpha=alpha,
        method='wilson'
    )
    
    return accuracy, (ci_lower, ci_upper)


def compute_accuracy_with_ci(data, condition=None, rung=None, category=None):
    """
    Compute accuracy with binomial CI, optionally filtered by condition/rung/category.
    Returns: (accuracy, ci_tuple, n_correct, n_total)
    """
    correct = 0
    total = 0
    
    for uuid, item in data.items():
        meta = item['metadata']
        response = item['response']
        gt = meta['ground_truth']
        
        # apply filters
        if condition and item['condition'] != condition:
            continue
        if rung is not None and meta['rung'] != rung:
            continue
        if category and meta['category'] != category:
            continue
        
        # skip if missing data
        if not response or not gt:
            continue
        
        total += 1
        if response == gt:
            correct += 1
    
    acc, ci = binomial_ci(correct, total)
    return acc, ci, correct, total


# =============================================================================
# BOOTSTRAP (only for metrics that need it - deltas and patterns)
# =============================================================================

def resample_by_questions(by_question, questions):
    """
    Bootstrap resample by selecting questions with replacement.
    Each question brings all its conditions along (preserves within-question structure).
    """
    # sample questions with replacement
    sampled_qids = np.random.choice(questions, size=len(questions), replace=True)
    
    resampled = {}
    counter = 0
    for qid in sampled_qids:
        for uuid, item in by_question[qid]:
            resampled[f"{uuid}_resample_{counter}"] = item
            counter += 1
    
    return resampled


def bootstrap_metric(data, metric_func, n_bootstrap=10000):
    """
    Bootstrap a metric by resampling questions.
    Returns: (mean, ci_lower, ci_upper, se)
    """
    # Build index once before loop
    by_question = defaultdict(list)
    for uuid, item in data.items():
        qid = item['metadata']['question_id']
        by_question[qid].append((uuid, item))
    
    questions = list(by_question.keys())
    
    bootstrap_values = []
    for i in range(n_bootstrap):
        if (i + 1) % 2000 == 0:
            print(f"    Bootstrap iteration {i+1}/{n_bootstrap}...", end='\r')
        
        resampled = resample_by_questions(by_question, questions)
        value = metric_func(resampled)
        bootstrap_values.append(value)
    
    print(f"    Completed {n_bootstrap} iterations.     ")
    
    # compute statistics
    values = np.array(bootstrap_values)
    mean = np.mean(values)
    se = np.std(values)
    ci_lower = np.percentile(values, 2.5)
    ci_upper = np.percentile(values, 97.5)
    
    return mean, ci_lower, ci_upper, se


def compute_delta_1(data):
    """Δ₁ = accuracy(original_graph) - accuracy(no_graph)"""
    acc_original, _, _, _ = compute_accuracy_with_ci(data, condition='original_graph')
    acc_no_graph, _, _, _ = compute_accuracy_with_ci(data, condition='no_graph')
    return acc_original - acc_no_graph


def compute_delta_2_avg(data):
    """
    Average Δ₂ across all corruption types. (w/ Bootstrap CI)
    """
    acc_original, _, _, _ = compute_accuracy_with_ci(data, condition='original_graph')
    
    deltas = []
    for corruption_type in CORRUPTION_TYPES:
        acc_corrupted, _, _, _ = compute_accuracy_with_ci(data, condition=corruption_type)
        if acc_original is not None and acc_corrupted is not None:
            deltas.append(acc_original - acc_corrupted)
    
    return np.mean(deltas) if deltas else 0

def compute_delta_2_individual(data, corruption_type):
    """Δ₂ for a specific corruption type = accuracy(original_graph) - accuracy(corruption_type)"""
    acc_original, _, _, _ = compute_accuracy_with_ci(data, condition='original_graph')
    acc_corrupted, _, _, _ = compute_accuracy_with_ci(data, condition=corruption_type)
    return acc_original - acc_corrupted


def compute_delta_3_avg(data):
    """
    Average Δ₃ across all corruption types. (w/ Bootstrap CI)
    Δ₃ = accuracy(no_graph) - accuracy(corrupted_graph)
    """
    acc_no_graph, _, _, _ = compute_accuracy_with_ci(data, condition='no_graph')
    
    deltas = []
    for corruption_type in CORRUPTION_TYPES:
        acc_corrupted, _, _, _ = compute_accuracy_with_ci(data, condition=corruption_type)
        if acc_no_graph is not None and acc_corrupted is not None:
            deltas.append(acc_no_graph - acc_corrupted)
    
    return np.mean(deltas) if deltas else 0


def compute_delta_3_individual(data, corruption_type):
    """Δ₃ for a specific corruption type = accuracy(no_graph) - accuracy(corruption_type)"""
    acc_no_graph, _, _, _ = compute_accuracy_with_ci(data, condition='no_graph')
    acc_corrupted, _, _, _ = compute_accuracy_with_ci(data, condition=corruption_type)
    return acc_no_graph - acc_corrupted


def compute_graph_benefit_rate(data):
    """
    Proportion of questions where graph helps (patterns 010, 011).
    Pattern 010: wrong without graph, correct with original, wrong with corrupted
    Pattern 011: wrong without graph, correct with both graphs
    """
    # group by question
    by_question = defaultdict(lambda: {
        'no_graph': None,
        'original_graph': None,
        'corrupted': None
    })
    
    for uuid, item in data.items():
        meta = item['metadata']
        response = item['response']
        gt = meta['ground_truth']
        qid = meta['question_id']
        condition = item['condition']
        
        if not response or not gt:
            continue
        
        is_correct = (response == gt)
        
        # map to simplified categories (see gen_patterns.py)
        if condition == 'no_graph':
            by_question[qid]['no_graph'] = is_correct
        elif condition == 'original_graph':
            by_question[qid]['original_graph'] = is_correct
        elif condition in CORRUPTION_TYPES:
            # OR aggregation: if correct on ANY corruption, consider it correct  (see gen_patterns.py)
            if by_question[qid]['corrupted'] is None:
                by_question[qid]['corrupted'] = is_correct
            else:
                by_question[qid]['corrupted'] = by_question[qid]['corrupted'] or is_correct
    
    # count pattern 010 + 011
    beneficial = 0
    total = 0
    
    for qid, results in by_question.items():
        ng = results['no_graph']
        og = results['original_graph']
        cg = results['corrupted']
        
        # need all three conditions
        if ng is None or og is None or cg is None:
            continue
        
        total += 1
        
        # pattern 010: only original graph correct
        if not ng and og and not cg:
            beneficial += 1
        # pattern 011: original and corrupted correct (but not no graph)
        elif not ng and og and cg:
            beneficial += 1
    
    return beneficial / total if total > 0 else 0


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_model(metadata, responses, model_name, n_bootstrap):
    """Run complete analysis with appropriate CI method for each metric."""
    
    responses_orig = responses.copy()
    responses = filter_valid_data(responses, metadata)
    
    invalid_count = len(responses_orig) - len(responses)
    if invalid_count > 0:
        print(f"  Filtered out {invalid_count} invalid responses")
    

    print(f"\nAnalyzing model: {model_name}")
    print("=" * 80)
    
    # Compute and store derived condition for each data point
    data = {}
    for uuid, meta in metadata.items():
        condition = get_condition(meta['prompt_type'], meta['corruption_type'])
        data[uuid] = {
            'metadata': meta,
            'response': responses.get(uuid, ""),
            'condition': condition
        }
    
    # count questions
    n_questions = len(get_unique_questions(data))
    print(f"Total questions: {n_questions}")
    print(f"Total responses: {len(data)}")
    
    results = {'model': model_name}
    
    # -------------------------------------------------------------------------
    # SECTION 1: Accuracy by condition (binomial CI - exact and simple)
    # -------------------------------------------------------------------------
    print("\n1. Computing accuracy by condition (binomial CI)...")
    results['by_condition'] = {}
    
    for condition in CONDITIONS:
        acc, ci, n_correct, n_total = compute_accuracy_with_ci(data, condition=condition)
        results['by_condition'][condition] = {
            'accuracy': acc,
            'ci': ci,
            'n_correct': n_correct,
            'n_total': n_total
        }
        print(f"   {condition}: {acc:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
    
    # -------------------------------------------------------------------------
    # SECTION 2: Deltas (bootstrap - needed for difference metrics)
    # -------------------------------------------------------------------------
    print("\n2. Computing delta metrics (bootstrap)...")
    results['deltas'] = {}
    
    print("   Δ₁ (original - no_graph)...")
    mean, ci_low, ci_high, se = bootstrap_metric(data, compute_delta_1, n_bootstrap)
    results['deltas']['delta_1'] = {
        'mean': mean,
        'ci': (ci_low, ci_high),
        'se': se
    }
    
    # Delta_2 AVERAGE with proper bootstrap CI (for overall hypothesis test)
    print("   Δ₂_avg (original - average of all corruptions)...")
    mean, ci_low, ci_high, se = bootstrap_metric(data, compute_delta_2_avg, n_bootstrap)
    results['deltas']['delta_2_avg'] = {
        'mean': mean,
        'ci': (ci_low, ci_high),
        'se': se
    }
    print(f"      Average: {mean:+.4f} [{ci_low:+.4f}, {ci_high:+.4f}]")
    
    # Individual delta_2 for each corruption type (with full bootstrap CIs)
    print("   Δ₂ individual (original - each corruption type)...")
    results['deltas']['delta_2_individual'] = {}
    
    for corruption_type in CORRUPTION_TYPES:
        print(f"      - {corruption_type}...")
        
        # Create function that captures this specific corruption_type
        def compute_func(d, ct=corruption_type):
            return compute_delta_2_individual(d, ct)
        
        mean, ci_low, ci_high, se = bootstrap_metric(data, compute_func, n_bootstrap)
        results['deltas']['delta_2_individual'][corruption_type] = {
            'mean': mean,
            'ci': (ci_low, ci_high),
            'se': se
        }
    
    # Delta_3 AVERAGE with proper bootstrap CI (for overall hypothesis test)
    print("   Δ₃_avg (no_graph - average of all corruptions)...")
    mean, ci_low, ci_high, se = bootstrap_metric(data, compute_delta_3_avg, n_bootstrap)
    results['deltas']['delta_3_avg'] = {
        'mean': mean,
        'ci': (ci_low, ci_high),
        'se': se
    }
    print(f"      Average: {mean:+.4f} [{ci_low:+.4f}, {ci_high:+.4f}]")
    
    # Individual delta_3 for each corruption type (with full bootstrap CIs)
    print("   Δ₃ individual (no_graph - each corruption type)...")
    results['deltas']['delta_3_individual'] = {}
    
    for corruption_type in CORRUPTION_TYPES:
        print(f"      - {corruption_type}...")
        
        # Create function that captures this specific corruption_type
        def compute_func(d, ct=corruption_type):
            return compute_delta_3_individual(d, ct)
        
        mean, ci_low, ci_high, se = bootstrap_metric(data, compute_func, n_bootstrap)
        results['deltas']['delta_3_individual'][corruption_type] = {
            'mean': mean,
            'ci': (ci_low, ci_high),
            'se': se
        }
    
    # -------------------------------------------------------------------------
    # SECTION 3: Accuracy by rung FOR ALL CONDITIONS (binomial CI)
    # -------------------------------------------------------------------------
    print("\n3. Computing accuracy by rung for all conditions (binomial CI)...")
    results['by_rung'] = {}
    
    for condition in CONDITIONS:
        print(f"\n   {condition}:")
        results['by_rung'][condition] = {}
        
        for rung in RUNGS:
            acc, ci, n_correct, n_total = compute_accuracy_with_ci(
                data, condition=condition, rung=rung
            )
            results['by_rung'][condition][rung] = {
                'accuracy': acc,
                'ci': ci,
                'n_correct': n_correct,
                'n_total': n_total
            }
            print(f"      Rung {rung}: {acc:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] ({n_correct}/{n_total})")
    
    # -------------------------------------------------------------------------
    # SECTION 4: Accuracy by sense category FOR ALL CONDITIONS (binomial CI)
    # -------------------------------------------------------------------------
    print("\n4. Computing accuracy by sense category for all conditions (binomial CI)...")
    results['by_category'] = {}
    
    for condition in CONDITIONS:
        print(f"\n   {condition}:")
        results['by_category'][condition] = {}
        
        for category in SENSE_CATEGORIES:
            acc, ci, n_correct, n_total = compute_accuracy_with_ci(
                data, condition=condition, category=category
            )
            results['by_category'][condition][category] = {
                'accuracy': acc,
                'ci': ci,
                'n_correct': n_correct,
                'n_total': n_total
            }
            print(f"      {category}: {acc:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] ({n_correct}/{n_total})")
    
    # -------------------------------------------------------------------------
    # SECTION 5: Graph benefit pattern (bootstrap - complex derived metric)
    # -------------------------------------------------------------------------
    print("\n5. Computing graph benefit rate (bootstrap)...")
    mean, ci_low, ci_high, se = bootstrap_metric(
        data, compute_graph_benefit_rate, n_bootstrap
    )
    results['graph_benefit'] = {
        'mean': mean,
        'ci': (ci_low, ci_high),
        'se': se
    }
    
    return results


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(results, output_path, n_bootstrap):
    """Generate text report with all CIs."""
    
    with output_path.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("CONFIDENCE INTERVAL REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"\nModel: {results['model']}\n")
        f.write("\nMethods:\n")
        f.write("  - Binomial CI (Wilson): for accuracy proportions (exact)\n")
        f.write(f"  - Bootstrap ({n_bootstrap} iterations): for deltas and patterns\n")
        f.write("\nInterpretation: 95% confidence intervals\n")
        f.write("  (If we sampled different questions, we're 95% confident\n")
        f.write("   the true value would fall in this range)\n")
        
        # Section 1: By condition
        f.write("\n" + "=" * 80 + "\n")
        f.write("ACCURACY BY CONDITION\n")
        f.write("=" * 80 + "\n\n")
        
        for condition in CONDITIONS:
            r = results['by_condition'][condition]
            f.write(f"{condition}:\n")
            f.write(f"  Accuracy: {r['accuracy']:.4f}\n")
            f.write(f"  95% CI:   [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]\n")
            f.write(f"  Sample:   {r['n_correct']}/{r['n_total']} correct\n")
            f.write("  Method:   Binomial (Wilson score interval)\n\n")
        
        # Section 2: Deltas
        f.write("\n" + "=" * 80 + "\n")
        f.write("DELTA METRICS (Effect of Graph)\n")
        f.write("=" * 80 + "\n\n")
        
        d1 = results['deltas']['delta_1']
        f.write("Δ₁ (original_graph - no_graph):\n")
        f.write(f"  Mean:    {d1['mean']:+.4f}\n")
        f.write(f"  95% CI:  [{d1['ci'][0]:+.4f}, {d1['ci'][1]:+.4f}]\n")
        f.write(f"  SE:      {d1['se']:.4f}\n")
        f.write("  Method:  Bootstrap (question-level resampling)\n")
        if d1['ci'][0] > 0:
            f.write("  Significantly positive: graph helps (p < 0.05)\n")
        elif d1['ci'][1] < 0:
            f.write("  Significantly negative: graph hurts (p < 0.05)\n")
        else:
            f.write("  Not significantly different from zero\n")
        f.write("\n")
        
        # Delta 2 - average with CI (primary metric)
        d2_avg = results['deltas']['delta_2_avg']
        f.write("Δ₂ (original_graph - corrupted_graphs):\n")
        f.write("\n  AVERAGE across all corruption types:\n")
        f.write(f"    Mean:    {d2_avg['mean']:+.4f}\n")
        f.write(f"    95% CI:  [{d2_avg['ci'][0]:+.4f}, {d2_avg['ci'][1]:+.4f}]\n")
        f.write(f"    SE:      {d2_avg['se']:.4f}\n")
        f.write("    Method:  Bootstrap (question-level resampling)\n")
        if d2_avg['ci'][0] > 0:
            f.write("    Significantly positive: corruption hurts on average (p < 0.05)\n")
        elif d2_avg['ci'][1] < 0:
            f.write("    Significantly negative: corruption helps on average!? (p < 0.05)\n")
        else:
            f.write("    Not significantly different from zero\n")
        
        # Individual corruption effects with full CIs
        f.write("\n  INDIVIDUAL corruption types:\n")
        for corruption_type in CORRUPTION_TYPES:
            d2 = results['deltas']['delta_2_individual'][corruption_type]
            f.write(f"\n    {corruption_type}:\n")
            f.write(f"      Mean:    {d2['mean']:+.4f}\n")
            f.write(f"      95% CI:  [{d2['ci'][0]:+.4f}, {d2['ci'][1]:+.4f}]\n")
            f.write(f"      SE:      {d2['se']:.4f}\n")
            if d2['ci'][0] > 0:
                f.write("      Significantly positive: this corruption hurts (p < 0.05)\n")
            elif d2['ci'][1] < 0:
                f.write("      Significantly negative: this corruption helps!? (p < 0.05)\n")
            else:
                f.write("      Not significantly different from zero\n")
        f.write("\n")
        
        # Delta 3 - average with CI
        d3_avg = results['deltas']['delta_3_avg']
        f.write("Δ₃ (no_graph - corrupted_graphs):\n")
        f.write("\n  AVERAGE across all corruption types:\n")
        f.write(f"    Mean:    {d3_avg['mean']:+.4f}\n")
        f.write(f"    95% CI:  [{d3_avg['ci'][0]:+.4f}, {d3_avg['ci'][1]:+.4f}]\n")
        f.write(f"    SE:      {d3_avg['se']:.4f}\n")
        f.write("    Method:  Bootstrap (question-level resampling)\n")
        if d3_avg['ci'][0] > 0:
            f.write("    Significantly positive: corruption hurts vs baseline (p < 0.05)\n")
        elif d3_avg['ci'][1] < 0:
            f.write("    Significantly negative: corruption helps vs baseline!? (p < 0.05)\n")
        else:
            f.write("    Not significantly different from zero\n")
        
        # Individual delta_3 effects with full CIs
        f.write("\n  INDIVIDUAL corruption types:\n")
        for corruption_type in CORRUPTION_TYPES:
            d3 = results['deltas']['delta_3_individual'][corruption_type]
            f.write(f"\n    {corruption_type}:\n")
            f.write(f"      Mean:    {d3['mean']:+.4f}\n")
            f.write(f"      95% CI:  [{d3['ci'][0]:+.4f}, {d3['ci'][1]:+.4f}]\n")
            f.write(f"      SE:      {d3['se']:.4f}\n")
            if d3['ci'][0] > 0:
                f.write("      Significantly positive: this corruption hurts vs baseline (p < 0.05)\n")
            elif d3['ci'][1] < 0:
                f.write("      Significantly negative: this corruption helps vs baseline!? (p < 0.05)\n")
            else:
                f.write("      Not significantly different from zero\n")
        f.write("\n")
        
        # Section 3: By rung (ALL CONDITIONS)
        f.write("\n" + "=" * 80 + "\n")
        f.write("ACCURACY BY CAUSAL RUNG (across all graph conditions)\n")
        f.write("=" * 80 + "\n\n")
        
        for condition in CONDITIONS:
            f.write(f"{condition}:\n")
            for rung in RUNGS:
                r = results['by_rung'][condition][rung]
                f.write(f"  Rung {rung}:\n")
                f.write(f"    Accuracy: {r['accuracy']:.4f}\n")
                f.write(f"    95% CI:   [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]\n")
                f.write(f"    Sample:   {r['n_correct']}/{r['n_total']} correct\n")
                f.write("    Method:   Binomial\n")
            f.write("\n")
        
        # Comparative evaluation for rungs
        f.write("COMPARATIVE EVAL (Rung-level):\n")
        f.write("-" * 80 + "\n")
        for rung in RUNGS:
            f.write(f"\nRung {rung} across conditions:\n")
            for condition in CONDITIONS:
                r = results['by_rung'][condition][rung]
                f.write(f"  {condition:20s}: {r['accuracy']:.4f} [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]\n")
        f.write("\n")
        
        # Section 4: By category (ALL CONDITIONS)
        f.write("\n" + "=" * 80 + "\n")
        f.write("ACCURACY BY SENSE CATEGORY (across all graph conditions)\n")
        f.write("=" * 80 + "\n\n")
        
        for condition in CONDITIONS:
            f.write(f"{condition}:\n")
            for category in SENSE_CATEGORIES:
                r = results['by_category'][condition][category]
                f.write(f"  {category.capitalize()}:\n")
                f.write(f"    Accuracy: {r['accuracy']:.4f}\n")
                f.write(f"    95% CI:   [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]\n")
                f.write(f"    Sample:   {r['n_correct']}/{r['n_total']} correct\n")
                f.write("    Method:   Binomial\n")
            f.write("\n")
        
        # Comparative evaluation for categories
        f.write("COMPARATIVE EVAL (Category-level):\n")
        f.write("-" * 80 + "\n")
        for category in SENSE_CATEGORIES:
            f.write(f"\n{category.capitalize()} across conditions:\n")
            for condition in CONDITIONS:
                r = results['by_category'][condition][category]
                f.write(f"  {condition:20s}: {r['accuracy']:.4f} [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]\n")
        f.write("\n")
        
        # Section 5: Graph benefit
        f.write("\n" + "=" * 80 + "\n")
        f.write("GRAPH BENEFIT PATTERN\n")
        f.write("=" * 80 + "\n\n")
        
        gb = results['graph_benefit']
        f.write("Proportion of questions where graph helps (patterns 010 + 011):\n")
        f.write(f"  Mean:    {gb['mean']:.4f}\n")
        f.write(f"  95% CI:  [{gb['ci'][0]:.4f}, {gb['ci'][1]:.4f}]\n")
        f.write(f"  SE:      {gb['se']:.4f}\n")
        f.write("  Method:  Bootstrap\n\n")
        f.write("Pattern 010: wrong with no graph, correct with original, wrong with corrupted\n")
        f.write("Pattern 011: wrong with no graph, correct with both original and corrupted\n")
        f.write("Note: 'corrupted' uses OR aggregation (correct on ANY corruption type)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute confidence intervals for causal reasoning evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Example:
        python gen_bootstrap_ci.py \\
            --metadata combined_metadata.jsonl \\
            --responses combined_responses.csv \\
            --model-name claude \\
            --n-bootstrap 10000 \\
            --output results/claude_ci.txt
        """
    )
    
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Combined metadata JSONL file"
    )
    
    parser.add_argument(
        "--responses",
        type=Path,
        required=True,
        help="Combined responses CSV file"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model being evaluated"
    )
    
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap iterations (default: 10000)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file path for CI report"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # set random seed
    np.random.seed(args.seed)
    
    # load data
    print(f"Loading metadata from {args.metadata}...")
    metadata = load_metadata(args.metadata)
    print(f"  Loaded {len(metadata)} entries")
    
    print(f"\nLoading responses from {args.responses}...")
    responses = load_responses(args.responses)
    print(f"  Loaded {len(responses)} responses")
    
    print_data_summary(metadata, responses)
    
    # filter valid data
    responses = filter_valid_data(responses, metadata)
    print(f"  Valid for analysis: {len(responses)}")
    
    # analyze (pass filtered responses)
    results = analyze_model(metadata, responses, args.model_name, args.n_bootstrap)
    
    # generate report
    print("\nGenerating report...")
    generate_report(results, args.output, args.n_bootstrap)
    print(f"\nReport saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())