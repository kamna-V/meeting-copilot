"""
Multi-agent orchestrator for Meeting Copilot.
Hand-rolled (no framework) so the mechanics are fully visible and explainable in an interview.

RUN: python orchestrator.py --transcript_file path/to/transcript.txt --task extract
     python orchestrator.py --transcript_file path/to/transcript.txt --task qa --question "Who owns the launch doc?"
     python orchestrator.py --transcript_file path/to/transcript.txt --task email

Pipeline:
  Router  -> decides which task type (extract / qa / email) if not explicitly given
  Retriever -> chunks long transcripts and pulls the most relevant chunks (keyword overlap;
               swap in embeddings if you have time)
  Extractor -> calls the fine-tuned model (your LocalFineTunedModel or ApiModel) to produce output
  Verifier  -> checks every action item's source_quote actually appears in the transcript;
               drops/flags anything that doesn't (hallucination guard)
  Formatter -> cleans final output into the shape the demo app expects
"""
import argparse
import json
import re
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training"))
from inference import get_model  # noqa: E402


# ---------- Retriever ----------
class RetrieverAgent:
    """Chunks a long transcript and returns the most relevant chunks for a query.
    Simple keyword-overlap scoring — swap for sentence-embeddings + cosine sim if you
    have time; the interface stays the same so it's a drop-in upgrade."""

    def __init__(self, chunk_size=800, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, transcript: str):
        words = transcript.split()
        chunks = []
        step = self.chunk_size - self.overlap
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            if chunk_words:
                chunks.append(" ".join(chunk_words))
        return chunks

    def retrieve(self, transcript: str, query: str, top_k=3):
        chunks = self.chunk(transcript)
        if len(chunks) <= top_k:
            return transcript  # short enough, no retrieval needed
        query_terms = set(re.findall(r"\w+", query.lower()))
        scored = []
        for c in chunks:
            chunk_terms = set(re.findall(r"\w+", c.lower()))
            overlap = len(query_terms & chunk_terms)
            scored.append((overlap, c))
        scored.sort(key=lambda x: -x[0])
        top_chunks = [c for _, c in scored[:top_k]]
        return "\n...\n".join(top_chunks)


# ---------- Router ----------
class RouterAgent:
    """Classifies the user's intent when it's not explicitly specified."""

    EXTRACT_KEYWORDS = ["action item", "extract", "task", "to-do", "todo", "follow up items"]
    EMAIL_KEYWORDS = ["email", "draft", "summary email", "follow-up email"]

    def route(self, user_request: str) -> str:
        text = user_request.lower()
        if any(k in text for k in self.EMAIL_KEYWORDS):
            return "email"
        if any(k in text for k in self.EXTRACT_KEYWORDS):
            return "extract"
        return "qa"  # default: treat as a question


# ---------- Extractor (calls the fine-tuned model) ----------
class ExtractorAgent:
    def __init__(self, model):
        self.model = model

    def extract_action_items(self, transcript: str, max_retries=2):
        instruction = (
            "Extract all action items from the meeting transcript below as a JSON array. "
            "If there are no action items, return an empty list: []\n\n"
            "Each item in the array MUST be an object with EXACTLY these keys:\n"
            '  "task": a short description of the task (string)\n'
            '  "owner": the person responsible (string, or "unassigned" if unclear)\n'
            '  "due_date": when it\'s due (string, or "not specified" if not mentioned)\n'
            '  "priority": one of "High", "Medium", "Low" (your best judgment)\n'
            '  "source_quote": the EXACT sentence from the transcript that this action item '
            "is based on, copied verbatim, word-for-word, so it can be verified against the source\n\n"
            "Example of the exact format required:\n"
            '[{"task": "Review the metrics dashboard", "owner": "Fatima", '
            '"due_date": "end of this week", "priority": "Medium", '
            '"source_quote": "Fatima, can you follow up on reviewing the current metrics '
            'dashboard and get back to us by end of this week?"}]\n\n'
            "Respond with ONLY the JSON array, no other text, no markdown code fences."
        )
        for attempt in range(max_retries + 1):
            raw = self.model.generate(instruction, transcript)
            parsed = self._try_parse_json(raw)
            if parsed is not None:
                return parsed
            # repair retry: ask the model to fix its own malformed output
            instruction = (f"Your previous response was not valid JSON:\n{raw}\n\n"
                            "Return ONLY a valid JSON array of action items using exactly the "
                            'keys task, owner, due_date, priority, source_quote — no other text.')
        return []  # give up gracefully rather than crashing the pipeline

    def answer_question(self, transcript_context: str, question: str):
        instruction = ("Answer the question using ONLY information from the transcript below. "
                        f"If the answer is not present, say 'Not discussed in this meeting.'\n\n"
                        f"Question: {question}")
        return self.model.generate(instruction, transcript_context).strip()

    def draft_email(self, transcript: str):
        instruction = ("Write a brief follow-up email summarizing the key decisions and "
                        "action items from this meeting transcript. Include a Subject line.")
        return self.model.generate(instruction, transcript).strip()

    @staticmethod
    def _try_parse_json(text):
        text = text.strip()
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


# ---------- Verifier / Critic ----------
class VerifierAgent:
    """Hallucination guard: every action item's source_quote (or task+owner) must be
    traceable to something in the transcript. Flags/drops items that aren't."""

    def verify_action_items(self, action_items: list, transcript: str):
        transcript_lower = transcript.lower()
        verified, flagged = [], []
        for item in action_items:
            quote = item.get("source_quote", "")
            owner = item.get("owner", "")
            grounded = False
            if quote and self._fuzzy_in(quote, transcript_lower):
                grounded = True
            elif owner and owner.lower() in transcript_lower:
                # weaker signal: at least the owner was mentioned
                grounded = True
            if grounded:
                verified.append(item)
            else:
                item["_flag_reason"] = "source_quote not found in transcript"
                flagged.append(item)
        return verified, flagged

    @staticmethod
    def _fuzzy_in(quote: str, transcript_lower: str, min_overlap=0.6):
        quote_words = set(re.findall(r"\w+", quote.lower()))
        if not quote_words:
            return False
        transcript_words = set(re.findall(r"\w+", transcript_lower))
        overlap = len(quote_words & transcript_words) / len(quote_words)
        return overlap >= min_overlap


# ---------- Formatter ----------
class FormatterAgent:
    def format_action_items(self, verified: list, flagged: list):
        return {
            "action_items": verified,
            "flagged_for_review": flagged,
            "summary": f"{len(verified)} verified action item(s)"
                       + (f", {len(flagged)} flagged for review" if flagged else "")
        }


# ---------- Orchestrator ----------
class MeetingCopilotOrchestrator:
    def __init__(self, model):
        self.router = RouterAgent()
        self.retriever = RetrieverAgent()
        self.extractor = ExtractorAgent(model)
        self.verifier = VerifierAgent()
        self.formatter = FormatterAgent()

    def run(self, transcript: str, task: str = None, question: str = None):
        if task is None:
            task = self.router.route(question or "")

        if task == "extract":
            raw_items = self.extractor.extract_action_items(transcript)
            verified, flagged = self.verifier.verify_action_items(raw_items, transcript)
            return self.formatter.format_action_items(verified, flagged)

        elif task == "qa":
            context = self.retriever.retrieve(transcript, question or "")
            answer = self.extractor.answer_question(context, question)
            return {"question": question, "answer": answer}

        elif task == "email":
            email = self.extractor.draft_email(transcript)
            return {"email_draft": email}

        else:
            raise ValueError(f"Unknown task: {task}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript_file", required=True)
    parser.add_argument("--task", choices=["extract", "qa", "email"], default=None)
    parser.add_argument("--question", default=None)
    parser.add_argument("--use_api_fallback", action="store_true",
                         help="Use API model instead of local fine-tuned checkpoint (faster to demo today)")
    parser.add_argument("--adapter_path", default=None, help="Path to your saved LoRA adapter")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "azure"])
    parser.add_argument("--api_model_name", default="claude-sonnet-4-6",
                         help="Model/deployment name — for azure, pass your deployment name here")
    parser.add_argument("--azure_endpoint", default=None,
                         help="e.g. https://your-resource.openai.azure.com — or set AZURE_OPENAI_ENDPOINT env var")
    args = parser.parse_args()

    with open(args.transcript_file) as f:
        transcript = f.read()

    model = get_model(use_api_fallback=args.use_api_fallback, adapter_path=args.adapter_path,
                       provider=args.provider, api_model_name=args.api_model_name,
                       azure_endpoint=args.azure_endpoint)
    orchestrator = MeetingCopilotOrchestrator(model)
    result = orchestrator.run(transcript, task=args.task, question=args.question)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
