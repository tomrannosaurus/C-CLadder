#!/usr/bin/env python3
"""Task 5: Statistical Analysis - accuracy by condition, scenario, rung.

Usage:
    python analysis/gen_stats.py \
        --metadata ./data/corruption/corrupted_causal_graphs_dataset_f1-3.jsonl \
        --data-dir ./data/model \
        --models gpt5:gpt5_responses_f1-3.csv \
        --output ./analysis/tables/_raw/gpt5_task5.txt

    python analysis/gen_stats.py \
        --metadata ./data/corruption/corrupted_causal_graphs_dataset_f1-3.jsonl \
        --data-dir ./data/model \
        --models claude:claude_responses_f1-3.csv \
        --output ./analysis/tables/_raw/claude_task5.txt
"""

import argparse
from pathlib import Path
import numpy as np
from analysis_common import (
    normalize_answer, 
    load_metadata_dict, 
    load_responses_dataframe,
    get_condition,
    print_data_summary,
    CORRUPTION_TYPES,
    CONDITIONS,
    SCENARIOS,
    RUNGS,
    RUNG_NAMES
)



def load_metadata(jsonl_path):
    """Load metadata from JSONL. Returns dict keyed by uuid."""
    return load_metadata_dict(jsonl_path)


def load_model_responses(csv_path, metadata=None):
    """Load responses from CSV. Returns DataFrame with normalized responses."""
    return load_responses_dataframe(csv_path, metadata)



def calc_accuracy(preds, truths):
    """Compute accuracy. Returns (accuracy, n)."""
    if len(preds) == 0:
        return None, 0
    preds = [normalize_answer(p) for p in preds]
    truths = [normalize_answer(t) for t in truths]
    correct = sum(p == t for p, t in zip(preds, truths))
    return correct / len(preds), len(preds)


def analyze(df, metadata):
    """Compute accuracy across all dimensions. Returns (results_dict, annotated_df)."""
    # Annotate dataframe
    df['ground_truth'] = df['uuid'].map(lambda x: metadata.get(x, {}).get('ground_truth', ''))
    df['prompt_type'] = df['uuid'].map(lambda x: metadata.get(x, {}).get('prompt_type', ''))
    df['corruption_type'] = df['uuid'].map(lambda x: metadata.get(x, {}).get('corruption_type'))
    df['condition'] = df.apply(lambda r: get_condition(r['prompt_type'], r['corruption_type']), axis=1)
    df['category'] = df['uuid'].map(lambda x: metadata.get(x, {}).get('category', ''))
    df['rung'] = df['uuid'].map(lambda x: metadata.get(x, {}).get('rung'))

    df_orig = df.copy()
    df = df[df['is_valid']].copy()
    
    invalid_count = len(df_orig) - len(df)
    if invalid_count > 0:
        print(f"  Filtered out {invalid_count} invalid responses")
    
    results = {'overall': {}, 'by_scenario': {}, 'by_rung': {}, 'errors': {}}

    # Overall by condition
    for cond in CONDITIONS:
        sub = df[df['condition'] == cond]
        if not sub.empty:
            acc, n = calc_accuracy(sub['response'].tolist(), sub['ground_truth'].tolist())
            results['overall'][cond] = {'accuracy': acc, 'n': n}
        else:
            results['overall'][cond] = {'accuracy': None, 'n': 0}

    # By scenario
    for scen in SCENARIOS:
        scen_df = df[df['category'] == scen]
        if scen_df.empty:
            continue
        results['by_scenario'][scen] = {}
        for cond in CONDITIONS:
            sub = scen_df[scen_df['condition'] == cond]
            if not sub.empty:
                acc, n = calc_accuracy(sub['response'].tolist(), sub['ground_truth'].tolist())
                results['by_scenario'][scen][cond] = {'accuracy': acc, 'n': n}
            else:
                results['by_scenario'][scen][cond] = {'accuracy': None, 'n': 0}

    # By rung
    for rung in RUNGS:
        rung_df = df[df['rung'] == rung]
        if rung_df.empty:
            continue
        results['by_rung'][rung] = {}
        for cond in CONDITIONS:
            sub = rung_df[rung_df['condition'] == cond]
            if not sub.empty:
                acc, n = calc_accuracy(sub['response'].tolist(), sub['ground_truth'].tolist())
                results['by_rung'][rung][cond] = {'accuracy': acc, 'n': n}
            else:
                results['by_rung'][rung][cond] = {'accuracy': None, 'n': 0}

    # Error patterns
    df['is_error'] = df['response'].str.lower() != df['ground_truth'].str.lower()
    for cond in CONDITIONS:
        sub = df[df['condition'] == cond]
        if sub.empty:
            continue
        err_df = sub[sub['is_error']]
        yes_no = len(err_df[(err_df['ground_truth'] == 'yes') & (err_df['response'] == 'no')])
        no_yes = len(err_df[(err_df['ground_truth'] == 'no') & (err_df['response'] == 'yes')])
        results['errors'][cond] = {
            'total': len(err_df),
            'yes_to_no': yes_no,
            'no_to_yes': no_yes,
            'other': len(err_df) - yes_no - no_yes
        }

    return results, df


def compute_deltas(results):
    """Compute delta metrics: graph benefit (delta1), corruption harm (delta2), baseline vs corruption (delta3)."""
    deltas = {}
    overall = results['overall']
    no_graph = overall['no_graph']['accuracy']
    correct = overall['original_graph']['accuracy']

    # Delta 1: correct graph vs no graph
    deltas['delta_1'] = (correct - no_graph) if (correct and no_graph) else None

    # Delta 2: correct graph vs each corruption
    d2_vals = []
    for corr in CORRUPTION_TYPES:
        corr_acc = overall[corr]['accuracy']
        if correct is not None and corr_acc is not None:
            d = correct - corr_acc
            deltas[f'delta_2_{corr}'] = d
            d2_vals.append(d)
        else:
            deltas[f'delta_2_{corr}'] = None

    deltas['avg_delta_2'] = np.mean(d2_vals) if d2_vals else None

    # Delta 3: no graph vs each corruption
    d3_vals = []
    for corr in CORRUPTION_TYPES:
        corr_acc = overall[corr]['accuracy']
        if no_graph is not None and corr_acc is not None:
            d = no_graph - corr_acc
            deltas[f'delta_3_{corr}'] = d
            d3_vals.append(d)
        else:
            deltas[f'delta_3_{corr}'] = None

    deltas['avg_delta_3'] = np.mean(d3_vals) if d3_vals else None

    return deltas


def avg_corruption_acc(results, level='overall', key=None):
    """Compute average accuracy across corruption types."""
    if level == 'overall':
        data = results['overall']
    elif level == 'by_scenario' and key in results.get('by_scenario', {}):
        data = results['by_scenario'][key]
    elif level == 'by_rung' and key in results.get('by_rung', {}):
        data = results['by_rung'][key]
    else:
        return None, 0

    accs, total_n = [], 0
    for corr in CORRUPTION_TYPES:
        if corr in data and data[corr]['accuracy'] is not None:
            accs.append(data[corr]['accuracy'])
            total_n += data[corr]['n']
    if accs:
        return np.mean(accs), total_n // len(accs)
    return None, 0


def generate_report(all_results, output_path):
    """Write text report."""
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("TASK 5: STATISTICAL ANALYSIS\n")
        f.write("=" * 70 + "\n")

        for model, data in all_results.items():
            res = data['results']
            deltas = data['deltas']

            f.write("\n" + "=" * 70 + "\n")
            f.write(f"MODEL: {model}\n")
            f.write("=" * 70 + "\n")

            # Section 1: Overall
            f.write("\nSECTION 1: OVERALL ACCURACY\n")
            f.write("-" * 70 + "\n")
            for cond in CONDITIONS:
                r = res['overall'][cond]
                if r['accuracy'] is not None:
                    f.write(f"  {cond:25s}: {r['accuracy']:.3f}  (n={r['n']})\n")
                else:
                    f.write(f"  {cond:25s}: N/A\n")

            # Summary
            ng = res['overall']['no_graph']['accuracy']
            og = res['overall']['original_graph']['accuracy']
            avg_corr, _ = avg_corruption_acc(res)

            f.write("\nSummary:\n")
            f.write(f"  No Graph:            {ng:.3f}\n" if ng else "  No Graph:            N/A\n")
            f.write(f"  Correct Graph:       {og:.3f}\n" if og else "  Correct Graph:       N/A\n")
            f.write(f"  Avg Corrupted:       {avg_corr:.3f}\n" if avg_corr else "  Avg Corrupted:       N/A\n")

            f.write("\nDeltas:\n")
            if deltas['delta_1'] is not None:
                f.write(f"  Delta1 (correct-none):     {deltas['delta_1']:+.3f}\n")
            if deltas['avg_delta_2'] is not None:
                f.write(f"  AvgDelta2 (correct-corr):  {deltas['avg_delta_2']:+.3f}\n")
            if deltas['avg_delta_3'] is not None:
                f.write(f"  AvgDelta3 (none-corr):     {deltas['avg_delta_3']:+.3f}\n")

            # Section 2: By Scenario
            f.write("\nSECTION 2: BY SCENARIO\n")
            f.write("-" * 70 + "\n")
            for scen in SCENARIOS:
                if scen not in res['by_scenario']:
                    continue
                f.write(f"\n{scen.upper()}:\n")
                scen_data = res['by_scenario'][scen]
                for cond in CONDITIONS:
                    r = scen_data[cond]
                    if r['accuracy'] is not None:
                        f.write(f"  {cond:25s}: {r['accuracy']:.3f}  (n={r['n']})\n")
                # Scenario summary
                s_ng = scen_data['no_graph']['accuracy']
                s_og = scen_data['original_graph']['accuracy']
                s_avg, _ = avg_corruption_acc(res, 'by_scenario', scen)
                f.write("\n  Summary:\n")
                if s_ng is not None: f.write(f"    No Graph:        {s_ng:.3f}\n")
                if s_og is not None: f.write(f"    Correct Graph:   {s_og:.3f}\n")
                if s_avg is not None: f.write(f"    Avg Corrupted:   {s_avg:.3f}\n")
                if s_ng is not None and s_og is not None:
                    f.write(f"  Delta1: {s_og - s_ng:+.3f}\n")
                if s_og is not None and s_avg is not None:
                    f.write(f"  AvgDelta2: {s_og - s_avg:+.3f}\n")
                if s_ng is not None and s_avg is not None:
                    f.write(f"  AvgDelta3: {s_ng - s_avg:+.3f}\n")

            # Section 3: By Rung
            f.write("\nSECTION 3: BY RUNG\n")
            f.write("-" * 70 + "\n")
            for rung in RUNGS:
                if rung not in res['by_rung']:
                    continue
                f.write(f"\nRung {rung} ({RUNG_NAMES[rung]}):\n")
                rung_data = res['by_rung'][rung]
                for cond in CONDITIONS:
                    r = rung_data[cond]
                    if r['accuracy'] is not None:
                        f.write(f"  {cond:25s}: {r['accuracy']:.3f}  (n={r['n']})\n")
                # Rung summary
                r_ng = rung_data['no_graph']['accuracy']
                r_og = rung_data['original_graph']['accuracy']
                r_avg, _ = avg_corruption_acc(res, 'by_rung', rung)
                f.write("\n  Summary:\n")
                if r_ng is not None: f.write(f"    No Graph:        {r_ng:.3f}\n")
                if r_og is not None: f.write(f"    Correct Graph:   {r_og:.3f}\n")
                if r_avg is not None: f.write(f"    Avg Corrupted:   {r_avg:.3f}\n")
                if r_ng is not None and r_og is not None:
                    f.write(f"  Delta1: {r_og - r_ng:+.3f}\n")
                if r_og is not None and r_avg is not None:
                    f.write(f"  AvgDelta2: {r_og - r_avg:+.3f}\n")
                if r_ng is not None and r_avg is not None:
                    f.write(f"  AvgDelta3: {r_ng - r_avg:+.3f}\n")

            # Section 4: Errors
            f.write("\nSECTION 4: ERROR PATTERNS\n")
            f.write("-" * 70 + "\n")
            for cond in CONDITIONS:
                if cond not in res['errors']:
                    continue
                err = res['errors'][cond]
                if err['total'] > 0:
                    f.write(f"\n{cond}:\n")
                    f.write(f"  Total: {err['total']}\n")
                    f.write(f"  Yes->No: {err['yes_to_no']} ({err['yes_to_no']/err['total']*100:.1f}%)\n")
                    f.write(f"  No->Yes: {err['no_to_yes']} ({err['no_to_yes']/err['total']*100:.1f}%)\n")
                    if err['other'] > 0:
                        f.write(f"  Other: {err['other']} ({err['other']/err['total']*100:.1f}%)\n")

            # Section 5: Research Questions
            f.write("\nSECTION 5: RESEARCH QUESTIONS\n")
            f.write("-" * 70 + "\n")

            # Q1: H0: correct graph has no effect
            f.write("\nQ1: Correct graph effect (H0: no effect)\n")
            if deltas['delta_1'] is not None:
                f.write(f"  Delta: {deltas['delta_1']:+.3f}\n")
                if abs(deltas['delta_1']) < 0.10:
                    f.write("  Note: Small delta; may be within margin of error\n")

            # Q2: H0: corruption has no effect
            f.write("\nQ2: Corruption effect (H0: no effect)\n")
            if deltas['avg_delta_2'] is not None:
                f.write(f"  Avg Delta: {deltas['avg_delta_2']:+.3f}\n")
                if abs(deltas['avg_delta_2']) < 0.10:
                    f.write("  Note: Small delta; may be within margin of error\n")
                for corr in CORRUPTION_TYPES:
                    d = deltas.get(f'delta_2_{corr}')
                    if d is not None:
                        f.write(f"    {corr:25s}: {d:+.3f}\n")

            # Q2b: H0: baseline vs corruption has no effect
            f.write("\nQ2b: Baseline vs corruption effect (H0: no effect)\n")
            if deltas['avg_delta_3'] is not None:
                f.write(f"  Avg Delta: {deltas['avg_delta_3']:+.3f}\n")
                if abs(deltas['avg_delta_3']) < 0.10:
                    f.write("  Note: Small delta; may be within margin of error\n")
                for corr in CORRUPTION_TYPES:
                    d = deltas.get(f'delta_3_{corr}')
                    if d is not None:
                        f.write(f"    {corr:25s}: {d:+.3f}\n")

            # Q3: H0: no scenario effect
            f.write("\nQ3: Scenario effect (H0: uniform across scenarios)\n")
            scen_accs = {s: res['by_scenario'].get(s, {}).get('original_graph', {}).get('accuracy')
                        for s in SCENARIOS}
            for s in SCENARIOS:
                if scen_accs.get(s) is not None:
                    f.write(f"  {s:20s}: {scen_accs[s]:.3f}\n")
            valid = [v for v in scen_accs.values() if v is not None]
            if len(valid) >= 2:
                range_val = max(valid) - min(valid)
                f.write(f"  Range: {range_val:.3f}\n")
                if range_val < 0.10:
                    f.write("  Note: Small range; differences may be within margin of error\n")

            # Q4: H0: no rung effect
            f.write("\nQ4: Rung effect (H0: uniform across rungs)\n")
            rung_accs = {r: res['by_rung'].get(r, {}).get('original_graph', {}).get('accuracy')
                        for r in RUNGS}
            for r in RUNGS:
                if rung_accs.get(r) is not None:
                    f.write(f"  Rung {r}: {rung_accs[r]:.3f}\n")
            valid = [v for v in rung_accs.values() if v is not None]
            if len(valid) >= 2:
                range_val = max(valid) - min(valid)
                f.write(f"  Range: {range_val:.3f}\n")
                if range_val < 0.10:
                    f.write("  Note: Small range; differences may be within margin of error\n")

            # Q5: H0: symmetric errors
            f.write("\nQ5: Error symmetry (H0: balanced yes->no vs no->yes)\n")
            tot_err = sum(res['errors'].get(c, {}).get('total', 0) for c in CONDITIONS)
            tot_yn = sum(res['errors'].get(c, {}).get('yes_to_no', 0) for c in CONDITIONS)
            tot_ny = sum(res['errors'].get(c, {}).get('no_to_yes', 0) for c in CONDITIONS)
            if tot_err > 0:
                pct_yn = tot_yn/tot_err*100
                pct_ny = tot_ny/tot_err*100
                f.write(f"  Yes->No: {tot_yn} ({pct_yn:.1f}%)\n")
                f.write(f"  No->Yes: {tot_ny} ({pct_ny:.1f}%)\n")
                if abs(pct_yn - pct_ny) < 10.0:
                    f.write("  Note: Small difference; may be within margin of error\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Task 5: Statistical analysis')
    parser.add_argument('--metadata', required=True, help='JSONL metadata file')
    parser.add_argument('--data-dir', default='./data/model', help='CSV directory')
    parser.add_argument('--models', nargs='+', required=True, help='name:file.csv pairs')
    parser.add_argument('--output-dir', default='./analysis', help='Output directory')
    parser.add_argument('--output', default=None, help='Output file (overrides --output-dir)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading metadata...")
    metadata = load_metadata(args.metadata)
    print(f"  {len(metadata)} entries")

    all_results = {}
    for spec in args.models:
        name, csv_file = spec.split(':')
        csv_path = Path(args.data_dir) / csv_file
        print(f"\nProcessing {name}...")

        if not csv_path.exists():
            print(f"  Error: {csv_path} not found")
            continue

        df = load_model_responses(csv_path, metadata)  
        print_data_summary(metadata, df) 
        
        results, _ = analyze(df, metadata)
        deltas = compute_deltas(results)
        all_results[name] = {'results': results, 'deltas': deltas}

        og_acc = results['overall']['original_graph']['accuracy']
        if og_acc:
            print(f"  Accuracy (correct graph): {og_acc:.3f}")

    if not all_results:
        print("\nNo models processed")
        return

    report_path = Path(args.output) if args.output else output_dir / 'task5_report.txt'
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nGenerating report...")
    generate_report(all_results, report_path)

    print("\n" + "=" * 50)
    print("COMPLETE")
    print("=" * 50)
    print(f"Report: {report_path}")


if __name__ == '__main__':
    main()