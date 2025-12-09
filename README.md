# Project Overview

This project investigates whether large language models integrate externally-provided causal graph constraints or primarily rely on parametric memory learned during pretraining. Through systematic graph corruption across different scenario types, we quantify the degree of structural reasoning versus pattern matching.

It was conducted as part of the WPI "Responsible AI" course (CS555) in Fall 2025.

## Core Research Question

To what extent do large language models integrate externally-provided causal graph constraints when these constraints conflict with internal causal priors, and which specific structural corruptions reliably induce reasoning failures?

# Experimental Design

## Six-Condition Framework
Each causal question is evaluated under six conditions:

1. **No Graph:** Question only (baseline)
2. **Correct Graph:**: Question with accurate causal graph
3. **Corruption 1:**: Random Edge Reversal
4. **Corruption 2:**: Collider (X → Collider ← Y)
5. **Corruption 3:**: Confounder (X ← Confounder → Y)
6. **Corruption 4:**: Mediator (X → Mediator → Y)

## Graph Corruption Mechanisms

Four specific corruption types applied to treatment variable X and outcome variable Y:

| Corruption Type | Implementation | Target Evaluation |
|-----------------|----------------|-------------------|
| Random Edge Reversal | Reverse one randomly selected edge | Causal direction understanding |
| Collider | Create X → Collider ← Y structure | Conditional independence reasoning |
| Confounder | Create X ← Confounder → Y structure | D-separation comprehension |
| Mediator | Create X → Mediator → Y path | Path-based reasoning |

## Dataset: CLadder Benchmark

- 504 stratified questions from CLadder's 10,000 question bank
- Three causal levels:
    - Rung 1: Association P(Y|X)
    - Rung 2: Intervention P(Y|do(X))
    - Rung 3: Counterfactual P(Y_x|X', Y')
- Three scenario types:
    - Commonsensical: Aligns with world knowledge
    - Anti-commonsensical: Contradicts common sense
    - Nonsensical: Abstract variables lacking semantic meaning

# Implementation

## Project Structure

````
causal_AI_proj/
├── analysis/        # Analysis scripts and results
│   └── tables/      # Generated tables for reporting
├── corruption/      # Graph corruption pipeline
├── data/            # Corrupted graph datasets and sampled questions
└── model/           # Model inference and API runner scripts
````

## Running the Corruption Pipeline

To generate corrupted causal graphs and corresponding prompts, run:

```bash
python ./corruption/corruption.py ./data/sampled_questions/sampled_questions_f1.json
```

```
Generate corrupted causal graphs and corresponding prompts.

positional arguments:
  question_file         Path to the questions JSON file.

options:
  -h, --help            show this help message and exit
  --meta_file META_FILE
                        Path to the meta models JSON file.
  --output_file OUTPUT_FILE
                        Path to the output JSONL file which contains the models responses. Default: data/corruption/corrupted_causal_graphs_dataset_f1.jsonl
```

The output file will be appended on the go. It is in [JSONL](https://jsonlines.org/) format.

## Running the local models

To run the local models on the corrupted dataset and export the responses to CSV, use the following commands:

```bash
python model/run_llm_and_export_csv.py 
    --input_jsonl data/corruption/corrupted_causal_graphs_dataset_f1.jsonl \
    --output_csv mistral.csv \
    --model_id mistralai/Mistral-7B-Instruct-v0.3 \
    --batch_size 16 # Speed Up
```

## Running API Models

*Note: Set your API keys as environment variables before running:*
```bash
export OPENAI_API_KEY="your-openai-key"
```
```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
```

To run the API models on the corrupted dataset and export the responses to CSV, use the following commands:
```bash
python model/run_api_and_export_csv.py \
    --provider openai \
    --input_jsonl data/corruption/corrupted_causal_graphs_dataset_f1.jsonl \
    --output_csv model/gpt5_mini_responses.csv
```
```bash
python model/run_api_and_export_csv.py \
    --provider anthropic \
    --input_jsonl data/corruption/corrupted_causal_graphs_dataset_f1.jsonl \
    --output_csv model/claude_sonnet4_responses.csv
```

## Analysis

### Combine Batches

#### Combine Metadata
```bash
python analysis/combine_batch.py \
        --metadata data/corruption/corrupted_causal_graphs_dataset_f1.jsonl data/corruption/corrupted_causal_graphs_dataset_f2.jsonl data/corruption/corrupted_causal_graphs_dataset_f3.jsonl \
        --output-metadata data/corruption/corrupted_causal_graphs_dataset_f1-3.jsonl
```

#### Combine Model Responses 
```bash
python analysis/combine_batch.py \
    --responses data/model/claude_responses_f1.csv data/model/claude_responses_f2.csv data/model/claude_responses_f3.csv \
    --output-responses data/model/claude_responses_f1-3.csv
```

### Generate Statistical Analysis
GPT-5:
```bash
python analysis/gen_stats.py \
    --metadata ./data/corruption/corrupted_causal_graphs_dataset_f1-3.jsonl \
    --data-dir ./data/model \
    --models gpt5:gpt5_responses_f1-3.csv \
    --output ./analysis/tables/_raw/gpt5_task5.txt
```

Claude:
```bash
python analysis/gen_stats.py \
    --metadata ./data/corruption/corrupted_causal_graphs_dataset_f1-3.jsonl \
    --data-dir ./data/model \
    --models claude:claude_responses_f1-3.csv \
    --output ./analysis/tables/_raw/claude_task5.txt
```

### Generate Pattern Analysis
GPT-5:
```bash
python analysis/gen_patterns.py \
    --metadata ./data/corruption/corrupted_causal_graphs_dataset_f1-3.jsonl \
    --data-dir ./data/model \
    --models gpt5:gpt5_responses_f1-3.csv \
    --output ./analysis/tables/_raw/gpt5_task6.txt
```

Claude:
```bash
python analysis/gen_patterns.py \
    --metadata ./data/corruption/corrupted_causal_graphs_dataset_f1-3.jsonl \
    --data-dir ./data/model \
    --models claude:claude_responses_f1-3.csv \
    --output ./analysis/tables/_raw/claude_task6.txt
```

### Generate Confidence Intervals
GPT-5:
```bash
python analysis/gen_ci.py \
    --metadata ./data/corruption/corrupted_causal_graphs_dataset_f1-3.jsonl \
    --responses ./data/model/gpt5_responses_f1-3.csv \
    --model-name gpt5 \
    --output ./analysis/tables/_raw/gpt5_ci.txt
```

Claude:
```bash
python analysis/gen_ci.py \
    --metadata ./data/corruption/corrupted_causal_graphs_dataset_f1-3.jsonl \
    --responses ./data/model/claude_responses_f1-3.csv \
    --model-name claude \
    --output ./analysis/tables/_raw/claude_ci.txt
```

# References
- Jin et al. (2024). CLadder: A Benchmark to Assess Causal Reasoning Capabilities of Language Models. NeurIPS 2024.

