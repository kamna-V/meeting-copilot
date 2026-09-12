"""
Gradio demo for Meeting Copilot.

RUN (using your real fine-tuned checkpoint, local inference):
    pip install gradio torch transformers peft accelerate
    python app.py

RUN (using Azure/OpenAI/Anthropic API instead — faster on a laptop with no GPU):
    Set USE_API_FALLBACK = True below, set PROVIDER/API_MODEL_NAME, then:
    pip install gradio openai
    export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
    export AZURE_OPENAI_API_KEY="your-key"
    python app.py

Then open the local URL Gradio prints (e.g. http://127.0.0.1:7860).
"""
import json
import os
import sys

import gradio as gr

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "agents"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))
from orchestrator import MeetingCopilotOrchestrator  # noqa: E402
from inference import get_model  # noqa: E402

USE_API_FALLBACK = False  # False = use your real fine-tuned checkpoint (local inference)
ADAPTER_PATH = "../training/checkpoints/v1_r16"  # path to your downloaded/unzipped checkpoint
PROVIDER = "azure"        # only used if USE_API_FALLBACK = True
API_MODEL_NAME = "YOUR-AZURE-DEPLOYMENT-NAME"  # only used if USE_API_FALLBACK = True

_model = None
_orchestrator = None


def get_orchestrator():
    global _model, _orchestrator
    if _orchestrator is None:
        _model = get_model(use_api_fallback=USE_API_FALLBACK, adapter_path=ADAPTER_PATH,
                            provider=PROVIDER, api_model_name=API_MODEL_NAME)
        _orchestrator = MeetingCopilotOrchestrator(_model)
    return _orchestrator


def load_transcript_file(filepath):
    """Reads an uploaded .txt transcript file into the transcript textbox."""
    if filepath is None:
        return gr.update()
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def run_extract(transcript):
    if not transcript.strip():
        return "Please paste a transcript first."
    orch = get_orchestrator()
    result = orch.run(transcript, task="extract")
    lines = [f"**{result['summary']}**\n"]
    for item in result["action_items"]:
        lines.append(f"- **{item.get('task') or '(untitled task)'}** — "
                      f"{item.get('owner') or 'unassigned'} "
                      f"(due {item.get('due_date') or 'not specified'})")
    if result["flagged_for_review"]:
        lines.append("\n**Flagged (could not verify against transcript):**")
        for item in result["flagged_for_review"]:
            lines.append(f"- ~~{item.get('task') or '(untitled task)'}~~ "
                          f"({item.get('_flag_reason', 'unverified')})")
    return "\n".join(lines)


def run_qa(transcript, question):
    if not transcript.strip() or not question.strip():
        return "Please provide both a transcript and a question."
    orch = get_orchestrator()
    result = orch.run(transcript, task="qa", question=question)
    return result["answer"]


def run_email(transcript):
    if not transcript.strip():
        return "Please paste a transcript first."
    orch = get_orchestrator()
    result = orch.run(transcript, task="email")
    return result["email_draft"]


SAMPLE_TRANSCRIPT = """Priya: Let's kick off — the pricing strategy overhaul is the main thing on the agenda today.
Fatima: Sounds good. I think we should revisit the enterprise tier first.
Priya: Agreed. Fatima, can you follow up on reviewing the current metrics dashboard and get back to us by end of this week?
Fatima: Sure, I'll have that ready.
Tom: What about the self-serve tier?
Priya: Let's have Tom handle drafting the updated proposal doc, target next Monday.
Tom: Got it, I'll get started today.
Priya: Great, thanks everyone, let's wrap up there."""

with gr.Blocks(title="Meeting Copilot") as demo:
    gr.Markdown("# Meeting Copilot\nFine-tuned LLM + multi-agent pipeline for meeting transcripts.")

    transcript_file = gr.File(label="Upload a transcript (.txt)", file_types=[".txt"])
    transcript_box = gr.Textbox(label="Meeting Transcript", lines=12, value=SAMPLE_TRANSCRIPT)
    transcript_file.upload(load_transcript_file, inputs=[transcript_file], outputs=[transcript_box])

    with gr.Tab("Extract Action Items"):
        extract_btn = gr.Button("Extract Action Items")
        extract_out = gr.Markdown()
        extract_btn.click(run_extract, inputs=[transcript_box], outputs=[extract_out])

    with gr.Tab("Ask a Question"):
        question_box = gr.Textbox(label="Question", placeholder="Who is responsible for the proposal?")
        qa_btn = gr.Button("Ask")
        qa_out = gr.Textbox(label="Answer")
        qa_btn.click(run_qa, inputs=[transcript_box, question_box], outputs=[qa_out])

    with gr.Tab("Draft Follow-up Email"):
        email_btn = gr.Button("Draft Email")
        email_out = gr.Textbox(label="Email Draft", lines=10)
        email_btn.click(run_email, inputs=[transcript_box], outputs=[email_out])

if __name__ == "__main__":
    demo.launch()
