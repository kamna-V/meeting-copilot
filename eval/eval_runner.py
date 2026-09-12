"""
End-to-end eval runner: runs the orchestrator over every gold-eval transcript,
scores the outputs with the LLM judge, and appends a row to a running
version-comparison table (results_log.csv) — this is your single strongest
artifact for the interview (Section 4.3 of the project plan).

RUN:
    python eval_runner.py --gold_file ../data/raw_records_gold_eval.json \
        --version_name "v1_finetuned_only" --use_api_fallback

    # after adding the verifier agent / retriever / etc, rerun with a new version_name:
    python eval_runner.py --gold_file ../data/raw_records_gold_eval.json \
        --version_name "v2_with_verifier" --use_api_fallback

Each run appends one row per (version_name) to results_log.csv with average scores,
so you build the comparison table automatically instead of copy-pasting numbers.
"""
import argparse
import csv
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "agents"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))
from orchestrator import MeetingCopilotOrchestrator  # noqa: E402
from inference import get_model  # noqa: E402
from judge import LLMJudge, evaluate_predictions  # noqa: E402


def run_pipeline_over_gold_set(gold_records, orchestrator):
    predictions = {}
    for rec in gold_records:
        extract_result = orchestrator.run(rec["transcript"], task="extract")
        email_result = orchestrator.run(rec["transcript"], task="email")
        predictions[rec["id"]] = {
            "action_items": extract_result["action_items"],
            "flagged_for_review": extract_result.get("flagged_for_review", []),
            "email": email_result["email_draft"],
        }
    return predictions


def summarize(judge_results):
    dims = ["correctness", "groundedness", "completeness"]
    summary = {}
    for dim in dims:
        vals = [r[dim] for r in judge_results if r.get(dim) is not None]
        summary[dim] = round(sum(vals) / len(vals), 2) if vals else None
    return summary


def append_to_log(version_name, summary, notes, log_path="results_log.csv"):
    file_exists = os.path.isfile(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["version", "correctness", "groundedness", "completeness", "notes"])
        writer.writerow([version_name, summary["correctness"], summary["groundedness"],
                          summary["completeness"], notes])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_file", required=True)
    parser.add_argument("--version_name", required=True,
                         help="e.g. v0_base_model, v1_finetuned, v2_with_verifier")
    parser.add_argument("--notes", default="")
    parser.add_argument("--use_api_fallback", action="store_true")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "azure"])
    parser.add_argument("--api_model_name", default="claude-sonnet-4-6",
                         help="Model/deployment name for the pipeline model — for azure, pass your deployment name")
    parser.add_argument("--azure_endpoint", default=None)
    parser.add_argument("--judge_provider", default="anthropic", choices=["anthropic", "openai", "azure"])
    parser.add_argument("--judge_model", default="claude-sonnet-4-6",
                         help="Model/deployment name for the JUDGE — for azure, pass your deployment name")
    parser.add_argument("--judge_azure_endpoint", default=None)
    parser.add_argument("--log_path", default="results_log.csv")
    args = parser.parse_args()

    with open(args.gold_file) as f:
        gold_records = json.load(f)

    model = get_model(use_api_fallback=args.use_api_fallback, adapter_path=args.adapter_path,
                       provider=args.provider, api_model_name=args.api_model_name,
                       azure_endpoint=args.azure_endpoint)
    orchestrator = MeetingCopilotOrchestrator(model)

    print(f"Running pipeline over {len(gold_records)} gold records...")
    predictions = run_pipeline_over_gold_set(gold_records, orchestrator)

    with open(f"predictions_{args.version_name}.json", "w") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print("Scoring with LLM judge...")
    judge = LLMJudge(provider=args.judge_provider, model=args.judge_model,
                      azure_endpoint=args.judge_azure_endpoint,
                      azure_deployment=args.judge_model if args.judge_provider == "azure" else None)
    judge_results = evaluate_predictions(gold_records, predictions, judge)

    with open(f"judge_results_{args.version_name}.json", "w") as f:
        json.dump(judge_results, f, indent=2)

    summary = summarize(judge_results)
    print(f"Summary for {args.version_name}: {summary}")

    append_to_log(args.version_name, summary, args.notes, log_path=args.log_path)
    print(f"Appended to {args.log_path} — this is your version-comparison table for the writeup.")


if __name__ == "__main__":
    main()
