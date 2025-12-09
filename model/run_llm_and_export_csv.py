import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, Optional
from transformers import pipeline, AutoTokenizer
from huggingface_hub import snapshot_download
import datetime

def download_model(model_name: str):
    """
    Download the model from Hugging Face Hub.
    """
    snapshot_download(model_name)

def build_pipe(model_id: str,
               return_full_text: bool = False):
    """
    Building the text-generation pipeline.
    - return_full_text = False: Only return the generated new item instead of the instruction words.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    pipe = pipeline(
        "text-generation",
        model=model_id,
        dtype="auto",
        tokenizer=tokenizer,
        device_map="auto",
        return_full_text=return_full_text,
    )
    return pipe


def iter_jsonl(path: Path):
    """
    Read JSONL
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def clean_text(s: Optional[str]) -> str:
    """
    Cleaning
    """
    if s is None:
        return ""
    return str(s).strip()

def process_batch(
    batch_entries,
    batch_messages,
    writer: csv.DictWriter,
    args,
    pipe,):

    if not batch_entries:
        return

    try:
        outputs = pipe(
            batch_messages,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            batch_size=args.batch_size,
        )
    except Exception as e:

        for entry in batch_entries:
            row = {
                "uuid": entry.get("uuid"),
                "model_name": args.model_id,
                "response": f"[ERROR] {repr(e)}",
                "timestamp": datetime.datetime.now().isoformat(),
            }
            writer.writerow(row)

        batch_entries.clear()
        batch_messages.clear()
        return

    for entry, out in zip(batch_entries, outputs):
        if isinstance(out, list):
            if len(out) > 0 and isinstance(out[0], dict):
                out_obj = out[0]
            else:
                response_text = clean_text(str(out))
                row = {
                    "uuid": entry.get("uuid"),
                    "model_name": args.model_id,
                    "response": response_text,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
                writer.writerow(row)
                continue
        elif isinstance(out, dict):
            out_obj = out
        else:
            response_text = clean_text(str(out))
            row = {
                "uuid": entry.get("uuid"),
                "model_name": args.model_id,
                "response": response_text,
                "timestamp": datetime.datetime.now().isoformat(),
            }
            writer.writerow(row)
            continue

        response_text = clean_text(out_obj.get("generated_text", ""))

        row = {
            "uuid": entry.get("uuid"),
            "model_name": args.model_id,
            "response": response_text,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        writer.writerow(row)

    batch_entries.clear()
    batch_messages.clear()

def main():
    parser = argparse.ArgumentParser(description="Run local LLM on prompts (JSONL) and export to CSV.")
    parser.add_argument("--input_jsonl", type=str, default="corrupted_causal_graphs_dataset.jsonl",
                        help="Generated JSONL document by corruption.py")
    parser.add_argument("--output_csv", type=str, default="llm_responses.csv",
                        help="CSV path")
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3",
                        help="meta-llama/Llama-3.1-8B-Instruct or mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens")
    parser.add_argument("--do_sample", action="store_true",
                        help="Sample or not")
    parser.add_argument("--limit", type=int, default=None,
                        help="For testing several data")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for parallel processing") # default batchsize is 1, try larger on HPC
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_csv)

    download_model(args.model_id)

    pipe = build_pipe(args.model_id, return_full_text=False)


    fieldnames = [
        "uuid",
        "model_name",
        "response",
        "timestamp",
    ]

    count = 0
    with output_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        batch_entries = []
        batch_messages = []

        for entry in iter_jsonl(input_path):
            if args.limit is not None and count >= args.limit:
                break

            prompt: str = clean_text(entry.get("prompt"))

            if not prompt:
                count += 1
                row = {
                    "uuid": entry.get("uuid"),
                    "model_name": args.model_id,
                    "response": "[NO_PROMPT]",
                    "timestamp": datetime.datetime.now().isoformat(),
                }
                writer.writerow(row)
                if count % 50 == 0:
                    print(f"Processed {count} rows...")
                continue

            messages = [
                {"role": "system", "content": "You are an assistant that only answers with 'Yes' or 'No'. Never explain or add punctuation."},
                {"role": "user", "content": prompt},
            ]

            batch_entries.append(entry)
            batch_messages.append(messages)
            count += 1

            if len(batch_entries) >= args.batch_size:
                process_batch(batch_entries, batch_messages, writer, args, pipe)
                if count % 50 == 0:
                    print(f"Processed {count} rows...")

        if batch_entries:
            process_batch(batch_entries, batch_messages, writer, args, pipe)

    print(f"Done! Wrote {count if args.limit is None else min(count, args.limit)} rows to {output_path}")


if __name__ == "__main__":
    main()