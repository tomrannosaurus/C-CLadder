#!/bin/bash
#SBATCH -N 1                   # 
#SBATCH -n 32                   # 
#SBATCH --mem=64g               # 
#SBATCH -J "ResAI Corruptions"   # 
#SBATCH -p academic               # 
#SBATCH -t 12:00:00            # 
#SBATCH --gres=gpu:1           # 
#SBATCH -C "A30"         # 

# Check that parameter is provided
if [ -z "$1" ]; then
  echo "Usage: sbatch sbatch.sh <version_number>"
  exit 1
fi

module load python             # 
module load cuda12.6/toolkit/12.6.2          # 

cd ~/causal_AI_proj/        #
source ~/causal_AI_proj/venv/bin/activate  #
python ./model/run_llm_and_export_csv.py --input_jsonl ./data/corruption/corrupted_causal_graphs_dataset_f$1.jsonl --output_csv ./data/model/mistral_f$1.csv --model_id mistralai/Mistral-7B-Instruct-v0.3 --max_new_tokens 3 --batch_size 64
python ./model/run_llm_and_export_csv.py --input_jsonl ./data/corruption/corrupted_causal_graphs_dataset_f$1.jsonl --output_csv ./data/model/llama_f$1.csv --model_id meta-llama/Llama-3.1-8B-Instruct --max_new_tokens 3 --batch_size 64

#todo: update this to run complete pipeline (from dataset to experiment), currently incomplete
