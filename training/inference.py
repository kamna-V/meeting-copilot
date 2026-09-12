"""
Inference wrapper — loads the base model + (optionally) your LoRA adapter and
exposes a single generate() function used by the agents and eval scripts.

RUN ON: same GPU environment as training (Colab/Kaggle), or any machine with the
model + adapter downloaded, or swap in an API call for a fast CPU-only demo (see
`ApiModel` fallback class at the bottom — use this if you're out of time/GPU and
just need the multi-agent + eval parts working end-to-end for the demo).
"""
class LocalFineTunedModel:
    def __init__(self, base_model_name="microsoft/Phi-3.5-mini-instruct", adapter_path=None):
        # Imported lazily so this module can be imported (e.g. by the orchestrator/tests)
        # on a machine without torch/GPU installed — only needed when you actually
        # instantiate this class.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        self._torch = torch
        # Always load the tokenizer from the base model, not the adapter folder.
        # LoRA doesn't change the vocabulary, and some transformers/tokenizers
        # version combos write a tokenizer_config.json into the adapter save dir
        # with a `tokenizer_class` value the currently-installed transformers
        # doesn't recognize (raises "Tokenizer class ... does not exist"). Loading
        # from the base model sidesteps that entirely.
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # device_map="auto" is meant for GPU / multi-GPU setups where accelerate
        # can offload layers across devices. On a CPU-only machine it instead
        # creates "meta" placeholder tensors that never get properly materialized,
        # which later breaks when peft tries to attach the LoRA adapter to real
        # module keys (KeyError on a module name, or "meta parameter" warnings
        # during loading). Detect CPU-only and load plainly instead.
        has_gpu = torch.cuda.is_available()
        if has_gpu:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            print("No GPU detected — loading in fp32 on CPU (device_map disabled). "
                  "This will be noticeably slower per generation than on a GPU.")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            self.model.to("cpu")

        if adapter_path:
            self.model = self._load_adapter_robustly(self.model, adapter_path)
        self.model.eval()

    @staticmethod
    def _load_adapter_robustly(base_model, adapter_path):
        """
        Loads a LoRA adapter even if adapter_config.json contains fields from a
        newer/older `peft` version than what's currently installed (this happens
        whenever a notebook session gets reset between training and inference,
        which pulls in whatever peft version ships with the fresh environment).
        Strips any config keys the installed LoraConfig doesn't recognize, on a
        copied config file, so the original checkpoint files are never modified.
        """
        import inspect
        import json
        import os
        import shutil
        import tempfile
        from peft import LoraConfig, PeftModel

        config_path = os.path.join(adapter_path, "adapter_config.json")
        with open(config_path) as f:
            raw_config = json.load(f)

        accepted = set(inspect.signature(LoraConfig.__init__).parameters.keys())
        filtered_config = {k: v for k, v in raw_config.items() if k in accepted}
        dropped = set(raw_config) - set(filtered_config)
        if dropped:
            print(f"Note: installed peft doesn't recognize these adapter_config fields, "
                  f"dropping them: {dropped}")

        if not dropped:
            # Nothing to strip — load directly, no need for a temp copy.
            return PeftModel.from_pretrained(base_model, adapter_path)

        # Copy the adapter dir to a temp location with the cleaned config, so we
        # never mutate the original checkpoint on disk.
        tmp_dir = tempfile.mkdtemp()
        for fname in os.listdir(adapter_path):
            src = os.path.join(adapter_path, fname)
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(tmp_dir, fname))
        with open(os.path.join(tmp_dir, "adapter_config.json"), "w") as f:
            json.dump(filtered_config, f, indent=2)

        return PeftModel.from_pretrained(base_model, tmp_dir)

    def generate(self, instruction: str, input_text: str, max_new_tokens=400) -> str:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text.split("### Response:")[-1].strip()


class ApiModel:
    """
    Fallback: use a hosted model via API instead of a local fine-tune.
    Useful if you run out of time/GPU — lets you still demo the full multi-agent +
    eval pipeline today, and swap in the real fine-tuned model later.

    provider="anthropic": requires ANTHROPIC_API_KEY env var
    provider="openai":    requires OPENAI_API_KEY env var
    provider="azure":     requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY env vars,
                           and `model` should be your Azure *deployment name* (not "gpt-4.1")
    """
    def __init__(self, provider="anthropic", model="claude-sonnet-4-6",
                 azure_endpoint=None, azure_api_version="2024-08-01-preview"):
        self.provider = provider
        self.model = model
        if provider == "anthropic":
            import anthropic  # lazy import — only needed if this class is actually used
            self.client = anthropic.Anthropic()
        elif provider == "openai":
            import openai  # lazy import — only needed if this class is actually used
            self.client = openai.OpenAI()
        elif provider == "azure":
            import openai
            import os
            self.client = openai.AzureOpenAI(
                azure_endpoint=azure_endpoint or os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=azure_api_version,
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def generate(self, instruction: str, input_text: str, max_new_tokens=400) -> str:
        prompt = f"{instruction}\n\n---\n{input_text}"
        if self.provider == "anthropic":
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=max_new_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        else:  # openai and azure share the same chat.completions interface
            resp = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_new_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content


def get_model(use_api_fallback=False, adapter_path=None, provider="anthropic",
               api_model_name="claude-sonnet-4-6", azure_endpoint=None):
    """Central place to switch between local fine-tuned model and API fallback."""
    if use_api_fallback:
        return ApiModel(provider=provider, model=api_model_name, azure_endpoint=azure_endpoint)
    return LocalFineTunedModel(adapter_path=adapter_path)