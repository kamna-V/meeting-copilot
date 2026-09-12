# Meeting Copilot — Execution Guide (1-Day Build)

Everything below has been syntax-checked and the multi-agent logic has been
unit-tested end-to-end with a mock model (verified the Verifier agent correctly
catches a fabricated action item). You're executing real, working code — not
pseudocode.

## Folder structure
```
meeting-copilot/
├── data_generated/           # ready-to-use dataset (already built)
│   ├── raw_records_train.json / raw_records_val.json / raw_records_gold_eval.json
│   ├── sft_train.jsonl / sft_val.jsonl
│   └── README.md
├── training/
│   ├── train_qlora.py        # RUN ON: Colab/Kaggle GPU
│   └── inference.py          # model wrapper (local fine-tuned OR API fallback)
├── agents/
│   └── orchestrator.py       # Router / Retriever / Extractor / Verifier / Formatter
├── eval/
│   ├── judge.py              # LLM-as-judge scoring
│   └── eval_runner.py        # runs orchestrator + judge, builds results_log.csv
├── demo/
│   └── app.py                # Gradio demo
└── requirements.txt
```

## IMPORTANT — the time-saving decision to make right now

Training a real QLoRA adapter (Section A below) takes ~30–90 min of actual GPU
time plus Colab setup overhead. Given one day, you have two honest paths:

- **Path 1 (full story):** Do the real fine-tune (Section A), then use
  `--adapter_path` everywhere. This gives you the strongest interview story
  ("here's my actual fine-tuned checkpoint and the before/after numbers").
- **Path 2 (time-safe):** Use `--use_api_fallback` everywhere (Claude/GPT-4o via
  API instead of your own checkpoint) to get the multi-agent system + eval
  pipeline fully working and demoable in under an hour, then squeeze in the real
  fine-tune with whatever time is left. **This is what I'd actually do on a
  1-day clock** — get the full pipeline working end-to-end first, fine-tune
  second. A working system beats an unfinished fine-tune.

Both paths use the exact same code — just a flag (`--use_api_fallback`) — so you
aren't locked in either way.

---

## Section A — Fine-tuning (Colab/Kaggle GPU)

1. Open a new Colab notebook, set runtime to GPU (T4 is fine).
2. Upload `data_generated/sft_train.jsonl`, `sft_val.jsonl`, and `training/train_qlora.py`
   to the Colab file browser (or `git clone` if you push this to a repo first).
3. First cell:
```bash
!pip install -q transformers==4.44.2 peft==0.12.0 trl==0.9.6 bitsandbytes==0.43.3 accelerate==0.33.0 datasets
```
4. Run training:
```bash
!python train_qlora.py --train_file sft_train.jsonl --val_file sft_val.jsonl \
    --output_dir ./checkpoints/v1_r16 --lora_r 16 --epochs 3
```
5. **For the ablation story** (do these if time allows — each is ~10-20 min on a T4):
```bash
# rank sweep
!python train_qlora.py --output_dir ./checkpoints/r8  --lora_r 8  --train_file sft_train.jsonl --val_file sft_val.jsonl
!python train_qlora.py --output_dir ./checkpoints/r32 --lora_r 32 --train_file sft_train.jsonl --val_file sft_val.jsonl

# data-scale curve
!python train_qlora.py --output_dir ./checkpoints/n50  --train_subset 50  --train_file sft_train.jsonl --val_file sft_val.jsonl
!python train_qlora.py --output_dir ./checkpoints/n150 --train_subset 150 --train_file sft_train.jsonl --val_file sft_val.jsonl
```
6. Download the `checkpoints/` folder (or push to a HF Hub repo) — you'll point
   `--adapter_path` at this from `agents/orchestrator.py` and `eval/eval_runner.py`.

---

## Section B — Multi-agent system (runs anywhere, no GPU needed)

Install lightweight deps:
```bash
pip install anthropic gradio
export ANTHROPIC_API_KEY=sk-...
```

Test the orchestrator directly on a transcript:
```bash
cd agents
python orchestrator.py --transcript_file ../sample_transcript.txt --task extract --use_api_fallback
python orchestrator.py --transcript_file ../sample_transcript.txt --task qa --question "Who owns the proposal?" --use_api_fallback
python orchestrator.py --transcript_file ../sample_transcript.txt --task email --use_api_fallback
```
(A `sample_transcript.txt` is included — see bottom of this guide.)

Once you have a real adapter from Section A, swap the flag:
```bash
python orchestrator.py --transcript_file ../sample_transcript.txt --task extract \
    --adapter_path ../training/checkpoints/v1_r16
```
(Requires torch/transformers/peft/bitsandbytes installed in this environment too —
run this part on the same GPU machine, or download the small adapter to a local
GPU box if you have one.)

---

## Section C — Evaluation pipeline (your strongest artifact)

```bash
cd eval
export ANTHROPIC_API_KEY=sk-...

# baseline: no fine-tune, no verifier smarts beyond what's built in
python eval_runner.py --gold_file ../data_generated/raw_records_gold_eval.json \
    --version_name "v0_api_baseline" --notes "API model, full pipeline, no fine-tune" \
    --use_api_fallback

# after you have a real fine-tuned adapter:
python eval_runner.py --gold_file ../data_generated/raw_records_gold_eval.json \
    --version_name "v1_finetuned" --notes "QLoRA r16, 3 epochs" \
    --adapter_path ../training/checkpoints/v1_r16
```

Every run appends a row to `eval/results_log.csv` — open it after 2-3 runs and
you'll have your version-comparison table ready to paste into the README/demo.

**Judge calibration (do this — it's the differentiator):** hand-score 10-15 of
the outputs yourself (1-5 on correctness/groundedness/completeness), put them in
a dict like `human_labels = {("m0021", "extract"): {"correctness": 4, ...}, ...}`,
and call `compare_to_human_labels()` from `judge.py` to get the correlation
number. This is the single highest-value 15 minutes you can spend today.

---

## Section D — Demo app

```bash
cd demo
pip install gradio
export ANTHROPIC_API_KEY=sk-...
python app.py
```
Open the printed local URL. Three tabs: Extract Action Items / Ask a Question /
Draft Email, pre-loaded with a sample transcript so it's demoable in 10 seconds.

To use your real fine-tuned model instead of the API: edit `USE_API_FALLBACK = False`
and `ADAPTER_PATH = "../training/checkpoints/v1_r16"` near the top of `app.py`.

---

## Realistic 1-day time budget

| Time | Task |
|---|---|
| 0:00–0:30 | Skim data, read this guide, set up API key |
| 0:30–1:30 | Section B: get orchestrator working with `--use_api_fallback`, sanity check on 2-3 transcripts |
| 1:30–2:30 | Section C: run eval_runner once (v0 baseline), do the 15-min human judge calibration |
| 2:30–4:00 | Section A: kick off QLoRA training on Colab (let it run in background while you do Section D) |
| 4:00–5:00 | Section D: get the Gradio demo working end-to-end |
| 5:00–6:00 | Rerun eval_runner with the real fine-tuned adapter → v1 row in results_log.csv |
| 6:00–7:00 | Write README, screenshot the comparison table, record a 2-3 min demo walkthrough |
| 7:00–8:00 | Buffer / polish / fix whatever broke |

---

## sample_transcript.txt (save this alongside the folders above)
```
Priya: Let's kick off — the pricing strategy overhaul is the main thing on the agenda today.
Fatima: Sounds good. I think we should revisit the enterprise tier first.
Priya: Agreed. Fatima, can you follow up on reviewing the current metrics dashboard and get back to us by end of this week?
Fatima: Sure, I'll have that ready.
Tom: What about the self-serve tier?
Priya: Let's have Tom handle drafting the updated proposal doc, target next Monday.
Tom: Got it, I'll get started today.
Priya: Great, thanks everyone, let's wrap up there.
```
