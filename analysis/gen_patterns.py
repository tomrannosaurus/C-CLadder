#!/usr/bin/env python3
"""Task 6: Pattern Analysis - question-level correctness patterns and consistency.

Usage:
    python analysis/gen_patterns.py \
        --metadata ./data/corruption/corrupted_causal_graphs_dataset_f1-3.jsonl \
        --data-dir ./data/model \
        --models gpt5:gpt5_responses_f1-3.csv \
        --output ./analysis/tables/_raw/gpt5_task6.txt

    python analysis/gen_patterns.py \
        --metadata ./data/corruption/corrupted_causal_graphs_dataset_f1-3.jsonl \
        --data-dir ./data/model \
        --models claude:claude_responses_f1-3.csv \
        --output ./analysis/tables/_raw/claude_task6.txt
"""


import argparse
from pathlib import Path
from collections import defaultdict, Counter
from analysis_common import (
    PATTERN_NAMES,
    load_metadata_dict,
    load_responses_dict,
    filter_valid_data,
    print_data_summary
)







def load_metadata(jsonl_path):
    """Load metadata from JSONL. Returns dict keyed by uuid."""
    return load_metadata_dict(jsonl_path)

def load_model_responses(csv_path):
    """Load responses from CSV. Returns dict keyed by uuid."""
    return load_responses_dict(csv_path)


def analyze(responses, metadata):
    """Analyze question-level patterns. Returns results dict."""
    responses_orig = responses.copy()
    responses = filter_valid_data(responses, metadata)
    
    invalid_count = len(responses_orig) - len(responses)
    if invalid_count > 0:
        print(f"  Filtered out {invalid_count} invalid responses")
    
    # Group by question
    by_question = defaultdict(lambda: {"no_graph": None, "original_graph": None, "corrupted_graph": None})

    for uuid, response in responses.items():
        if uuid not in metadata:
            continue
        meta = metadata[uuid]
        qid = meta["question_id"]
        ptype = meta["prompt_type"]
        gt = meta["ground_truth"]

        if response == "" or gt == "": # This should never happen now, keeping for safety
            continue
        if ptype not in ("no_graph", "original_graph", "corrupted_graph"):
            continue

        by_question[qid][ptype] = {"pred": response, "gt": gt, "is_correct": (response == gt)}

    # Compute patterns
    results = {
        "pattern_counts": Counter(),
        "total_questions": 0,
        "with_all_three": 0,
        "consistent_all": 0,
        "change_when_add_graph": 0,
        "change_when_corrupted": 0,
        "question_details": [],
    }

    for qid, answers in by_question.items():
        ng = answers["no_graph"]
        og = answers["original_graph"]
        cg = answers["corrupted_graph"]

        has_ng = ng is not None
        has_og = og is not None
        has_cg = cg is not None

        if not (has_ng or has_og or has_cg):
            continue

        results["total_questions"] += 1

        if has_ng and has_og and has_cg:
            results["with_all_three"] += 1

        # Binary pattern
        b_ng = int(ng["is_correct"]) if has_ng else 0
        b_og = int(og["is_correct"]) if has_og else 0
        b_cg = int(cg["is_correct"]) if has_cg else 0
        pattern = f"{b_ng}{b_og}{b_cg}"
        results["pattern_counts"][pattern] += 1

        # Consistency (only if all three present)
        is_consistent = False
        changed_add = False
        changed_corrupt = False

        if has_ng and has_og and has_cg:
            pred_ng = ng["pred"]
            pred_og = og["pred"]
            pred_cg = cg["pred"]

            if pred_ng == pred_og == pred_cg:
                is_consistent = True
                results["consistent_all"] += 1
            if pred_ng != pred_og:
                changed_add = True
                results["change_when_add_graph"] += 1
            if pred_og != pred_cg:
                changed_corrupt = True
                results["change_when_corrupted"] += 1

        results["question_details"].append({
            "question_id": qid,
            "pattern_code": pattern,
            "pattern_name": PATTERN_NAMES.get(pattern, "unknown"),
            "has_all_three": has_ng and has_og and has_cg,
            "correct_ng": ng["is_correct"] if has_ng else None,
            "correct_og": og["is_correct"] if has_og else None,
            "correct_cg": cg["is_correct"] if has_cg else None,
            "pred_ng": ng["pred"] if has_ng else "",
            "pred_og": og["pred"] if has_og else "",
            "pred_cg": cg["pred"] if has_cg else "",
            "is_consistent": is_consistent,
            "changed_add_graph": changed_add,
            "changed_corrupted": changed_corrupt,
        })

    return results


def generate_report(all_results, output_path):
    """Write text report."""
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("TASK 6: PATTERN ANALYSIS\n")
        f.write("=" * 70 + "\n")

        for model, res in all_results.items():
            pattern_counts = res["pattern_counts"]
            total = res["total_questions"]
            with_all = res["with_all_three"]
            pattern_total = sum(pattern_counts.values())

            f.write("\n" + "=" * 70 + "\n")
            f.write(f"MODEL: {model}\n")
            f.write("=" * 70 + "\n")

            # Section 1: Overview
            f.write("\nSECTION 1: OVERVIEW\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Total questions:         {total}\n")
            f.write(f"  With all 3 conditions:   {with_all}\n")

            # Section 2: Pattern Distribution
            f.write("\nSECTION 2: PATTERN DISTRIBUTION\n")
            f.write("-" * 70 + "\n")
            f.write("  Pattern: [no_graph][original][corrupted]  (1=correct, 0=wrong)\n\n")
            for code in sorted(pattern_counts.keys()):
                count = pattern_counts[code]
                pct = (count / pattern_total * 100) if pattern_total > 0 else 0
                name = PATTERN_NAMES.get(code, "unknown")
                f.write(f"  {code} ({name:30s}): {count:4d}  ({pct:5.1f}%)\n")
            f.write(f"\n  Total: {pattern_total}\n")

            # Section 3: Consistency
            f.write("\nSECTION 3: CONSISTENCY\n")
            f.write("-" * 70 + "\n")
            if with_all > 0:
                consistent = res["consistent_all"]
                change_graph = res["change_when_add_graph"]
                change_corrupt = res["change_when_corrupted"]
                f.write(f"  (n={with_all} questions with all 3 conditions)\n\n")
                f.write(f"  Consistent (all same):      {consistent:4d}  ({consistent/with_all*100:5.1f}%)\n")
                f.write(f"  Changed adding graph:       {change_graph:4d}  ({change_graph/with_all*100:5.1f}%)\n")
                f.write(f"  Changed by corruption:      {change_corrupt:4d}  ({change_corrupt/with_all*100:5.1f}%)\n")
            else:
                f.write("  Insufficient data\n")

            # Section 4: Sensitivity Metrics
            f.write("\nSECTION 4: SENSITIVITY METRICS\n")
            f.write("-" * 70 + "\n")
            beneficial = pattern_counts.get("010", 0) + pattern_counts.get("011", 0)
            harmful = pattern_counts.get("100", 0) + pattern_counts.get("101", 0)
            corruption_sens = pattern_counts.get("110", 0)
            anomalous = pattern_counts.get("001", 0)

            f.write(f"  Graph beneficial (010,011):   {beneficial:4d}\n")
            f.write(f"  Graph harmful (100,101):      {harmful:4d}\n")
            f.write(f"  Corruption sensitive (110):   {corruption_sens:4d}\n")
            f.write(f"  Anomalous (001):              {anomalous:4d}\n")
            f.write(f"\n  Net graph effect:             {beneficial - harmful:+4d}\n")

            # Section 5: Research Questions
            f.write("\nSECTION 5: RESEARCH QUESTIONS\n")
            f.write("-" * 70 + "\n")

            # Q1: H0: model ignores graph
            f.write("\nQ1: Graph sensitivity (H0: model ignores graph)\n")
            if with_all > 0:
                sensitivity = 1 - (res["consistent_all"] / with_all)
                f.write(f"  Changed answers: {sensitivity*100:.1f}%\n")

            # Q2: H0: correct graph has no effect
            f.write("\nQ2: Correct graph effect (H0: no benefit)\n")
            f.write(f"  Helped (010,011): {beneficial}\n")
            f.write(f"  Hurt (100,101):   {harmful}\n")
            net_graph = beneficial - harmful
            f.write(f"  Net: {net_graph:+d}\n")
            if with_all > 0 and abs(net_graph) < max(10, with_all * 0.10):
                f.write("  Note: Small net effect; may be within margin of error\n")

            # Q3: H0: corruption has no effect
            f.write("\nQ3: Corruption effect (H0: no degradation)\n")
            f.write(f"  Hurt (110):       {corruption_sens}\n")
            f.write(f"  Helped (001):     {anomalous}\n")
            net_corrupt = corruption_sens - anomalous
            f.write(f"  Net: {net_corrupt:+d}\n")
            if with_all > 0 and abs(net_corrupt) < max(10, with_all * 0.10):
                f.write("  Note: Small net effect; may be within margin of error\n")

            # Q4: Difficulty distribution
            f.write("\nQ4: Difficulty distribution\n")
            always_correct = pattern_counts.get("111", 0)
            always_wrong = pattern_counts.get("000", 0)
            graph_dep = pattern_total - always_correct - always_wrong
            if pattern_total > 0:
                f.write(f"  Always correct (111): {always_correct:4d} ({always_correct/pattern_total*100:.1f}%)\n")
                f.write(f"  Always wrong (000):   {always_wrong:4d} ({always_wrong/pattern_total*100:.1f}%)\n")
                f.write(f"  Graph-dependent:      {graph_dep:4d} ({graph_dep/pattern_total*100:.1f}%)\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Task 6: Pattern analysis')
    parser.add_argument('--metadata', required=True, help='JSONL metadata file')
    parser.add_argument('--data-dir', default='./data/model', help='CSV directory')
    parser.add_argument('--models', nargs='+', required=True, help='name:file.csv pairs')
    parser.add_argument('--output-dir', default='./analysis', help='Output directory')
    parser.add_argument('--output', default=None, help='Output file (overrides --output-dir)')
    args = parser.parse_args()

    if args.output:
        report_path = Path(args.output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / 'task6_report.txt'

    print("Loading metadata...")
    metadata = load_metadata(Path(args.metadata))
    print(f"  {len(metadata)} entries")

    all_results = {}
    for spec in args.models:
        name, csv_file = spec.split(':')
        csv_path = Path(args.data_dir) / csv_file
        print(f"\nProcessing {name}...")

        if not csv_path.exists():
            print(f"  Error: {csv_path} not found")
            continue

        responses = load_model_responses(csv_path)
        print(f"  {len(responses)} responses")
        
        print_data_summary(metadata, responses)
        
        results = analyze(responses, metadata)
        all_results[name] = results

        print(f"  Total questions: {results['total_questions']}")
        print(f"  With all 3 conditions: {results['with_all_three']}")
        if results['with_all_three'] > 0:
            rate = results['consistent_all'] / results['with_all_three'] * 100
            print(f"  Consistency: {rate:.1f}%")

    if not all_results:
        print("\nNo models processed")
        return

    print("\nGenerating report...")
    generate_report(all_results, report_path)

    print("\n" + "=" * 50)
    print("COMPLETE")
    print("=" * 50)
    print(f"Report: {report_path}")


if __name__ == '__main__':
    main()