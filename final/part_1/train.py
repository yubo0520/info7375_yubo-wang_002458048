import os
import re
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig, PeftModel
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import think_format_reward


#  data

def make_prompt(target, nums):
    # qwen-instruct template from TinyZero countdown.py 
    return [
        {"role": "system", "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."},
        {"role": "user", "content": f" Using the numbers {nums}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."},
    ]


def make_dataset(n, offset=0):
    raw = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    raw = raw.select(range(offset, offset + n))
    records = []
    for ex in raw:
        records.append({
            "prompt": make_prompt(ex["target"], ex["nums"]),
            "target": ex["target"],
            "nums": ex["nums"],
        })
    return Dataset.from_list(records)


#  reward helpers

def _extract_answer(text):
    matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return matches[-1].strip() if matches else None


def _validate(eq_str, nums):
    # check that the equation uses exactly the provided numbers, each once
    try:
        used = sorted(int(n) for n in re.findall(r'\d+', eq_str))
        return used == sorted(int(n) for n in nums)
    except:
        return False


def _eval_eq(eq_str):
    # safely evaluate an arithmetic expression
    # strip trailing "= result" that models sometimes append
    if '=' in eq_str:
        eq_str = eq_str.split('=')[0].strip()
    try:
        if not re.match(r'^[\d+\-*/().\s]+$', eq_str):
            return None
        return eval(eq_str, {"__builtins__": None}, {})
    except:
        return None


#  reward functions

def countdown_accuracy_reward(completions, target, nums, **kwargs):
    # 1.0 if correct, 0.1 if right format but wrong answer, 0.0 if no answer
    # TRL 1.0 passes completions as list of message dicts for chat models
    rewards = []
    for comp, tgt, ns in zip(completions, target, nums):
        text = comp[-1]["content"] if isinstance(comp, list) else comp
        eq = _extract_answer(text)
        if eq is None:
            rewards.append(0.0)
            continue
        if not _validate(eq, ns):
            rewards.append(0.1)
            continue
        result = _eval_eq(eq)
        if result is not None and abs(result - tgt) < 1e-5:
            rewards.append(1.0)
        else:
            rewards.append(0.1)
    return rewards


#  eval

def evaluate(model, tokenizer, dataset, max_new_tokens=512):
    model.eval()
    correct = 0
    total = len(dataset)
    for ex in dataset:
        input_ids = tokenizer.apply_chat_template(
            ex["prompt"], tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )["input_ids"].to(model.device)
        # pad_token and eos_token can be identical for some instruct checkpoints; pass mask explicitly.
        attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        gen = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        eq = _extract_answer(gen)
        if eq and _validate(eq, ex["nums"]):
            result = _eval_eq(eq)
            if result is not None and abs(result - ex["target"]) < 1e-5:
                correct += 1
    return correct / total if total > 0 else 0.0


#  callback

class ShowExamplesCallback(TrainerCallback):
    def __init__(self, tokenizer, samples, every_n_steps=25):
        self.tokenizer = tokenizer
        self.samples = samples  # small list of dicts from eval_data
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 or state.global_step % self.every_n_steps != 0:
            return
        if model is None:
            return
        model.eval()
        print(f"\n--- examples at step {state.global_step} ---")
        for ex in self.samples[:2]:
            input_ids = self.tokenizer.apply_chat_template(
                ex["prompt"], tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )["input_ids"].to(model.device)
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=256,
                    do_sample=False, pad_token_id=self.tokenizer.eos_token_id
                )
            gen = self.tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
            eq = _extract_answer(gen)
            result = _eval_eq(eq) if eq else None
            ok = result is not None and abs(result - ex["target"]) < 1e-5
            print(f"  target={ex['target']}, nums={ex['nums']}")
            print(f"  answer: {eq}  result: {result}  correct: {ok}")
            print(f"  output: {gen[:300]}")
        model.train()


def main():
    model_name = "Qwen/Qwen2.5-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("loading dataset...")
    train_data = make_dataset(1000, offset=0)
    eval_data = make_dataset(100, offset=1000)
    print(f"  train: {len(train_data)}, eval: {len(eval_data)}")


    print("\nPhase 1: Baseline Evaluation")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    baseline_acc = evaluate(base_model, tokenizer, eval_data)
    print(f"Baseline accuracy: {baseline_acc:.1%}")
    del base_model
    torch.cuda.empty_cache()


    if os.path.exists("/runs"):
        output_dir = "/runs/countdown_lora_3b"
    elif os.path.exists("/content/drive/MyDrive"):
        output_dir = "/content/drive/MyDrive/countdown_lora_3b"
    else:
        output_dir = "./countdown_lora_3b"
    print(f"  output_dir: {output_dir}")

    print("\nPhase 2: GRPO Fine-Tuning with LoRA")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=5e-6,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        adam_beta1=0.9,
        adam_beta2=0.999,
        weight_decay=0.01,
        max_grad_norm=1.0,
        num_generations=4,
        max_completion_length=256,
        beta=0.001,
        logging_steps=5,
        bf16=True,
        save_strategy="steps",
        save_steps=50,
        report_to="none",
        reward_weights=[2.0, 1.0],  # accuracy weighted 2x over format
    )

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=[countdown_accuracy_reward, think_format_reward],
        args=training_args,
        train_dataset=train_data,
        peft_config=peft_config,
    )

    trainer.add_callback(ShowExamplesCallback(tokenizer, list(eval_data.select(range(3))), every_n_steps=25))
    has_ckpt = os.path.isdir(output_dir) and any(d.startswith("checkpoint-") for d in os.listdir(output_dir))
    trainer.train(resume_from_checkpoint=has_ckpt)
    del trainer
    torch.cuda.empty_cache()


    print("\nPhase 3: Evaluate All Checkpoints")

    ckpt_dirs = sorted(
        [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
        key=lambda d: int(d.split("-")[1]),
    )

    results = {}
    for ckpt_name in ckpt_dirs:
        ckpt_path = os.path.join(output_dir, ckpt_name)
        step = int(ckpt_name.split("-")[1])
        print(f"\nevaluating {ckpt_name}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, ckpt_path)
        model = model.merge_and_unload()
        acc = evaluate(model, tokenizer, eval_data)
        results[step] = acc
        print(f"  step {step}: {acc:.1%}")
        del model, base_model
        torch.cuda.empty_cache()

    if results:
        best_step = max(results, key=results.get)
        print(f"\nbest checkpoint: step {best_step}  acc: {results[best_step]:.1%}")
        print(f"all results: {results}")


if __name__ == "__main__":
    main()
