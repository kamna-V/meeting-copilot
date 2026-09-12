"""
QLoRA fine-tuning for Meeting Copilot.
RUN THIS ON: Google Colab (T4/A100 GPU) or Kaggle GPU notebook — NOT locally unless you have a GPU.

Setup on Colab, first cell:
    !pip install -q transformers==4.44.2 peft==0.12.0 trl==0.9.6 bitsandbytes==0.43.3 accelerate==0.33.0 datasets

Upload sft_train.jsonl and sft_val.jsonl to the Colab working directory (or mount Drive),
then run:
    python train_qlora.py --train_file sft_train.jsonl --val_file sft_val.jsonl --output_dir ./checkpoints/run1

Swap --model_name to try different bases/ablations, e.g.:
    microsoft/Phi-3.5-mini-instruct
    Qwen/Qwen2.5-7B-Instruct
"""
import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


def load_jsonl_as_dataset(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    texts = [PROMPT_TEMPLATE.format(**r) for r in rows]
    return Dataset.from_dict({"text": texts})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--train_file", default="sft_train.jsonl")
    parser.add_argument("--val_file", default="sft_val.jsonl")
    parser.add_argument("--output_dir", default="./checkpoints/run1")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--train_subset", type=int, default=None,
                         help="Use only N training examples — for the data-scale ablation curve")
    parser.add_argument("--save_steps", type=int, default=20,
                         help="Save a checkpoint every N steps — keep this low so an interrupted "
                              "session loses as little progress as possible")
    parser.add_argument("--resume", action="store_true",
                         help="Resume from the latest checkpoint found in --output_dir, if any. "
                              "Use this every time you restart after a Kaggle session dies.")
    args = parser.parse_args()

    print(f"Loading base model: {args.model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = load_jsonl_as_dataset(args.train_file)
    if args.train_subset:
        train_ds = train_ds.select(range(min(args.train_subset, len(train_ds))))
        print(f"Using data-scale subset: {len(train_ds)} examples")
    val_ds = load_jsonl_as_dataset(args.val_file)

    # Build training args in a version-tolerant way: newer trl versions want an
    # SFTConfig (which subclasses TrainingArguments but has a different/larger
    # kwarg set) rather than plain TrainingArguments, and different transformers
    # versions have dropped/renamed some kwargs (e.g. evaluation_strategy ->
    # eval_strategy). Rather than hardcoding one shape, we filter our desired
    # kwargs down to whatever the installed class actually accepts.
    desired_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        eval_strategy="epoch",
        evaluation_strategy="epoch",  # older transformers name; filtered below if unsupported
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=True,
        report_to="none",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
    )

    try:
        from trl import SFTConfig
        args_cls = SFTConfig
        print("Using trl.SFTConfig for training arguments.")
    except ImportError:
        args_cls = TrainingArguments
        print("Using transformers.TrainingArguments for training arguments.")

    import inspect
    accepted = set(inspect.signature(args_cls.__init__).parameters.keys())
    filtered_kwargs = {k: v for k, v in desired_kwargs.items() if k in accepted}
    dropped = set(desired_kwargs) - set(filtered_kwargs)
    if dropped:
        print(f"Note: this installed version doesn't accept these kwargs, skipping: {dropped}")

    training_args = args_cls(**filtered_kwargs)

    sft_trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer_accepted = set(inspect.signature(SFTTrainer.__init__).parameters.keys())
    if "dataset_text_field" in trainer_accepted:
        sft_trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in trainer_accepted:
        sft_trainer_kwargs["max_seq_length"] = args.max_seq_len
    # Prefer processing_class (current transformers Trainer API) over the older
    # `tokenizer` kwarg — some trl versions still *accept* `tokenizer` in their own
    # signature but then forward it into a newer transformers.Trainer that no
    # longer supports it, causing a TypeError. Checking processing_class first
    # avoids that mismatch whenever it's available.
    if "processing_class" in trainer_accepted:
        sft_trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_accepted:
        sft_trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**sft_trainer_kwargs)

    resume_from = None
    if args.resume:
        import glob
        checkpoints = sorted(
            glob.glob(os.path.join(args.output_dir, "checkpoint-*")),
            key=lambda p: int(p.split("-")[-1])
        )
        if checkpoints:
            resume_from = checkpoints[-1]
            print(f"Resuming from checkpoint: {resume_from}")
        else:
            print("No checkpoint found in output_dir yet — starting fresh (this is normal on first run).")

    trainer.train(resume_from_checkpoint=resume_from)

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
