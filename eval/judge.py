"""
LLM-as-Judge evaluation pipeline for Meeting Copilot.

Scores model/pipeline outputs on Correctness, Groundedness, Completeness (and Tone
for emails), using a strong external model as judge (Claude or GPT-4o via API).

RUN: requires ANTHROPIC_API_KEY (or OPENAI_API_KEY) set in your environment.
    export ANTHROPIC_API_KEY=sk-...
    python judge.py --gold_file ../data/raw_records_gold_eval.json --predictions_file predictions.json

`predictions.json` should be a list of {id, predicted_action_items, predicted_answers,
predicted_email} produced by running the orchestrator over each gold record — see
eval_runner.py, which does this end-to-end for you.
"""
import argparse
import json
import os
import re


JUDGE_PROMPT_TEMPLATE = """You are grading an AI meeting-assistant's output against a transcript.
Score strictly. Respond with ONLY a JSON object, no other text.

TRANSCRIPT:
{transcript}

TASK: {task_description}

MODEL OUTPUT:
{model_output}

Score the output on these dimensions (integers 1-5, 5 = best):
- correctness: does the output accurately reflect what was said?
- groundedness: is every claim traceable to the transcript (no hallucination)?
- completeness: did it capture the relevant information without missing things?

Respond with exactly this JSON shape:
{{"correctness": <int 1-5>, "groundedness": <int 1-5>, "completeness": <int 1-5>, "reasoning": "<one sentence>"}}
"""


class LLMJudge:
    def __init__(self, provider="anthropic", model="claude-sonnet-4-6",
                 azure_endpoint=None, azure_deployment=None, azure_api_version="2024-08-01-preview"):
        self.provider = provider
        self.model = model
        if provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic()
        elif provider == "openai":
            import openai
            self.client = openai.OpenAI()
        elif provider == "azure":
            import openai
            # Reads AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY from env if not passed explicitly.
            self.client = openai.AzureOpenAI(
                azure_endpoint=azure_endpoint or os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=azure_api_version,
            )
            self.model = azure_deployment or model  # Azure calls this the "deployment name"
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def score(self, transcript: str, task_description: str, model_output: str) -> dict:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            transcript=transcript, task_description=task_description, model_output=model_output
        )
        if self.provider == "anthropic":
            resp = self.client.messages.create(
                model=self.model, max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text
        else:  # openai and azure share the same chat.completions interface
            resp = self.client.chat.completions.create(
                model=self.model, max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content

        return self._parse_json(raw)

    @staticmethod
    def _parse_json(text: str) -> dict:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {"correctness": None, "groundedness": None, "completeness": None,
                     "reasoning": "PARSE_ERROR", "raw": text}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"correctness": None, "groundedness": None, "completeness": None,
                     "reasoning": "PARSE_ERROR", "raw": text}


def evaluate_predictions(gold_records: list, predictions: dict, judge: LLMJudge) -> list:
    """
    gold_records: list of records from raw_records_gold_eval.json
    predictions: dict keyed by record id -> {"action_items": [...], "email": "..."}
    Returns a list of per-record judge scores.
    """
    results = []
    for rec in gold_records:
        pred = predictions.get(rec["id"])
        if pred is None:
            continue
        transcript = rec["transcript"]

        if "action_items" in pred:
            score = judge.score(
                transcript,
                "Extract all action items as a list of {task, owner, due_date}.",
                json.dumps(pred["action_items"]),
            )
            results.append({"id": rec["id"], "task": "extract", **score})

        if "email" in pred:
            score = judge.score(
                transcript,
                "Write a follow-up email summarizing decisions and action items.",
                pred["email"],
            )
            results.append({"id": rec["id"], "task": "email", **score})

    return results


def compare_to_human_labels(judge_results: list, human_labels: dict):
    """
    Correlate judge scores against your own hand-assigned scores (1-5) for the same
    records, stored in human_labels: {(id, task): {"correctness": int, ...}}.
    Requires numpy/scipy — falls back to a simple mean-diff if unavailable.
    """
    try:
        from scipy.stats import spearmanr
        have_scipy = True
    except ImportError:
        have_scipy = False

    for dim in ["correctness", "groundedness", "completeness"]:
        judge_vals, human_vals = [], []
        for r in judge_results:
            key = (r["id"], r["task"])
            if key in human_labels and r.get(dim) is not None:
                judge_vals.append(r[dim])
                human_vals.append(human_labels[key][dim])
        if len(judge_vals) < 3:
            print(f"{dim}: not enough overlapping labeled examples to correlate")
            continue
        if have_scipy:
            corr, _ = spearmanr(judge_vals, human_vals)
            print(f"{dim}: Spearman correlation = {corr:.2f}  (n={len(judge_vals)})")
        else:
            mean_diff = sum(abs(j - h) for j, h in zip(judge_vals, human_vals)) / len(judge_vals)
            print(f"{dim}: mean abs diff = {mean_diff:.2f}  (n={len(judge_vals)}) [install scipy for correlation]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_file", required=True)
    parser.add_argument("--predictions_file", required=True)
    parser.add_argument("--output_file", default="judge_results.json")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "azure"])
    parser.add_argument("--model", default="claude-sonnet-4-6",
                         help="Model name (anthropic/openai) — ignored for azure, use --azure_deployment instead")
    parser.add_argument("--azure_endpoint", default=None,
                         help="e.g. https://your-resource.openai.azure.com — or set AZURE_OPENAI_ENDPOINT env var")
    parser.add_argument("--azure_deployment", default=None,
                         help="Your Azure deployment name for the gpt-4.1 model")
    args = parser.parse_args()

    with open(args.gold_file) as f:
        gold_records = json.load(f)
    with open(args.predictions_file) as f:
        predictions = json.load(f)

    judge = LLMJudge(provider=args.provider, model=args.model,
                      azure_endpoint=args.azure_endpoint, azure_deployment=args.azure_deployment)
    results = evaluate_predictions(gold_records, predictions, judge)

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    avg_correctness = sum(r["correctness"] for r in results if r.get("correctness")) / max(
        1, len([r for r in results if r.get("correctness")]))
    print(f"Average correctness across {len(results)} scored outputs: {avg_correctness:.2f}/5")
    print(f"Full results saved to {args.output_file}")


if __name__ == "__main__":
    main()
