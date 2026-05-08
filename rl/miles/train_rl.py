import os
import sys
import json
import time
import asyncio
import subprocess

import requests
import httpx
import numpy as np
from transformers import AutoTokenizer
import pathlib
from sglang.utils import wait_for_server
from huggingface_hub import snapshot_download
import subprocess, time, socket
import openai
from gsm8k_utils import load_gsm8k, GSM8KDataLoader
from tqdm.asyncio import tqdm as atqdm
from reward_utils import grade_answer_verl
import matplotlib.pyplot as plt
from gsm8k_utils import format_prompt


# Download Qwen3-0.6B if not already present
MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/tiger/models/Qwen3-0.6B")

if not os.path.exists(MODEL_PATH):
    print(f"Downloading Qwen3-0.6B to {MODEL_PATH}...")
    snapshot_download(repo_id="Qwen/Qwen3-0.6B", local_dir=MODEL_PATH)
    print("Download complete!")
else:
    print(f"Model already exists at {MODEL_PATH}")

# SGLang Inference Engine
SGLANG_HOST = "127.0.0.1"
SGLANG_PORT = 30000
SGLANG_URL = f"http://{SGLANG_HOST}:{SGLANG_PORT}"

# FSDP Training Engine
TRAIN_HOST = "127.0.0.1"
TRAIN_PORT = 5000
TRAIN_URL = f"http://{TRAIN_HOST}:{TRAIN_PORT}"

# RL Hyperparameters
BATCH_SIZE = 16               # Unique prompts per rollout iteration
N_SAMPLES_PER_PROMPT = 8      # Responses per prompt (for GRPO group normalization)
ROLLOUT_TEMPERATURE = 1.0     # Sampling temperature for diversity
MAX_NEW_TOKENS = 1024         # Max response length
LR = 5e-6                     # Learning rate
EPS_CLIP = 0.2                # PPO clipping epsilon
NUM_ITERATIONS = 40           # Number of training iterations

# Micro-batching: split the rollout batch into smaller chunks for the forward/backward pass.
# This trades iteration time for peak GPU memory. The training result is identical to
# processing the full batch at once (gradients are accumulated and scaled correctly).
# Rule of thumb: MICRO_BATCH_SIZE × MAX_NEW_TOKENS ≲ 16k tokens per micro-batch.
MICRO_BATCH_SIZE = 4          # Samples per micro-batch (BATCH_SIZE*N_SAMPLES / MICRO_BATCH_SIZE micro-batches)

# SGLang TP size (for weight sync)
TP_SIZE = 2

print(f"Model: {MODEL_PATH}")
print(f"SGLang URL: {SGLANG_URL}")
print(f"Training URL: {TRAIN_URL}")
print(f"Batch size: {BATCH_SIZE} prompts × {N_SAMPLES_PER_PROMPT} samples = {BATCH_SIZE * N_SAMPLES_PER_PROMPT} total")
print(f"Micro-batch size: {MICRO_BATCH_SIZE} ({BATCH_SIZE * N_SAMPLES_PER_PROMPT // MICRO_BATCH_SIZE} micro-batches per step)")


SGLANG_CMD = (
    f"python3 -m sglang.launch_server "
    f"--model-path {MODEL_PATH} "
    f"--tp {TP_SIZE} "
    f"--host {SGLANG_HOST} --port {SGLANG_PORT} "
    f"--mem-fraction-static 0.3 "
    f"--enable-memory-saver "
    f"--trust-remote-code"
)
print(f"Launch command:\n{SGLANG_CMD}")


def _port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def _kill_servers():
    for pattern in ["sglang", "fsdp_training_server"]:
        subprocess.run(f"pkill -9 -f {pattern} || true", shell=True, stderr=subprocess.DEVNULL)

    for port in [SGLANG_PORT, TRAIN_PORT]:
        subprocess.run(
            f"lsof -ti:{port} | xargs -r kill -9 || true",
            shell=True, stderr=subprocess.DEVNULL
        )

    time.sleep(5)

_kill_servers()


_log_dir = pathlib.Path("logs")
_log_dir.mkdir(exist_ok=True)
_sglang_log_path = _log_dir / "sglang_server.log"
_sglang_log_file = open(_sglang_log_path, "w")

sglang_proc = subprocess.Popen(
    SGLANG_CMD,
    shell=True,
    stdout=_sglang_log_file,
    stderr=_sglang_log_file,
    text=True,
)
print(f"SGLang server log: {_sglang_log_path.resolve()}")
wait_for_server(SGLANG_URL, timeout=300)
print("SGLang server is ready!")


client = openai.Client(base_url=f"{SGLANG_URL}/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model=MODEL_PATH,
    messages=[{"role": "user", "content": "What is 2+2?"}],
    temperature=0,
    max_tokens=64,
)
print(f"SGLang response: {response.choices[0].message.content}")


TRAIN_CMD = (
    f"torchrun --nproc-per-node={TP_SIZE} "
    f"fsdp_training_server.py "
    f"--model-path {MODEL_PATH} "
    f"--sglang-url {SGLANG_URL} "
    f"--port {TRAIN_PORT} "
    f"--lr {LR} "
    f"--eps-clip {EPS_CLIP} "
    f"--tp-size {TP_SIZE} "
    f"--micro-batch-size {MICRO_BATCH_SIZE} "
    f"--gradient-checkpointing"
)
print(f"Launch command:\n{TRAIN_CMD}")


_train_log_path = _log_dir / "train_server.log"
_train_log_file = open(_train_log_path, "w")

train_proc = subprocess.Popen(
    TRAIN_CMD,
    shell=True,
    stdout=_train_log_file,
    stderr=_train_log_file,
    text=True,
    cwd=os.path.dirname(os.path.abspath("fsdp_training_server.py")),
)
print(f"Training server log: {_train_log_path.resolve()}")

# Wait for training server to be ready
for i in range(120):
    try:
        resp = requests.get(f"{TRAIN_URL}/health", timeout=5)
        if resp.status_code == 200:
            print(f"Training server is ready! {resp.json()}")
            break
    except Exception:
        pass
    time.sleep(2)
    if i % 10 == 0:
        print(f"Waiting for training server... ({i*2}s)")
else:
    print("ERROR: Training server failed to start. Check logs:")
    print(train_proc.stdout.read())


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
train_data = load_gsm8k(split="train")
test_data = load_gsm8k(split="test")

print(f"Train: {len(train_data)} problems")
print(f"Test:  {len(test_data)} problems")
print(f"\nExample:")
print(f"  Question: {train_data[0]['question'][:200]}...")
print(f"  Answer:   {train_data[0]['label']}")

data_loader = GSM8KDataLoader(
    train_data,
    tokenizer,
    batch_size=BATCH_SIZE,
    n_samples_per_prompt=N_SAMPLES_PER_PROMPT,
)
print(f"DataLoader: {len(data_loader)} prompts, batch_size={BATCH_SIZE}, n_samples={N_SAMPLES_PER_PROMPT}")


async def rollout_batch(sglang_url, batch, temperature, max_new_tokens):
    """Generate responses for a batch of prompt groups.

    Mirrors miles/rollout/sglang_rollout.py: generate().

    Args:
        sglang_url: SGLang server URL
        batch: list of groups, each group is a list of sample dicts
        temperature: sampling temperature
        max_new_tokens: max response length

    Returns:
        list of response dicts with keys:
            full_token_ids, rollout_log_probs, response_text,
            prompt_length, response_length, label
    """
    results = []

    async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
        tasks = []
        sample_meta = []  # Track metadata for each request

        for group in batch:
            for sample in group:
                payload = {
                    "input_ids": sample["token_ids"],
                    "sampling_params": {
                        "temperature": temperature,
                        "max_new_tokens": max_new_tokens,
                        "skip_special_tokens": True,
                    },
                    "return_logprob": True,
                }
                tasks.append(client.post(f"{sglang_url}/generate", json=payload))
                sample_meta.append(sample)

        # Execute all requests concurrently with progress bar
        responses = await atqdm.gather(*tasks, desc="Rollout", total=len(tasks))

    for resp, sample in zip(responses, sample_meta):
        output = resp.json()
        meta = output.get("meta_info", {})

        # Extract log probs for response tokens
        # SGLang returns: output_token_logprobs = [(logprob, token_id, token_text), ...]
        output_logprobs = meta.get("output_token_logprobs", [])
        rollout_log_probs = [item[0] for item in output_logprobs]

        # Extract response token IDs
        response_token_ids = [item[1] for item in output_logprobs]

        results.append({
            "full_token_ids": sample["token_ids"] + response_token_ids,
            "rollout_log_probs": rollout_log_probs,
            "response_text": output.get("text", ""),
            "prompt_length": len(sample["token_ids"]),
            "response_length": len(response_token_ids),
            "label": sample["label"],
        })

    return results


# Test rollout with a single batch
batch = data_loader.get_batch()
# responses = await rollout_batch(SGLANG_URL, batch, ROLLOUT_TEMPERATURE, MAX_NEW_TOKENS)
responses = asyncio.run(rollout_batch(SGLANG_URL, batch, ROLLOUT_TEMPERATURE, MAX_NEW_TOKENS))

print(f"Generated {len(responses)} responses")
print(f"\nExample response:")
print(f"  Prompt length:   {responses[0]['prompt_length']} tokens")
print(f"  Response length: {responses[0]['response_length']} tokens")
print(f"  Text: {responses[0]['response_text'][:300]}...")
print(f"  Log probs (first 5): {responses[0]['rollout_log_probs'][:5]}")


# Compute rewards for all responses
rewards = []
for r in responses:
    is_correct = grade_answer_verl(r["response_text"], r["label"])
    rewards.append(1.0 if is_correct else 0.0)

mean_reward = sum(rewards) / len(rewards)
print(f"Mean reward: {mean_reward:.3f} ({sum(r == 1.0 for r in rewards)}/{len(rewards)} correct)")


def filter_zero_std_groups(responses, rewards, n_samples_per_prompt):
    """Filter out prompt groups where all rewards are identical (std = 0).

    Mirrors miles/rollout/filter_hub/dynamic_sampling_filters.py: check_reward_nonzero_std().

    When every response for a prompt has the same reward, GRPO group-normalization
    produces zero advantage for every token in that group → zero gradient signal.
    Removing these groups before the training step avoids wasting forward/backward
    passes on examples that contribute nothing to learning.

    Common zero-std cases:
        • All N_SAMPLES responses correct  (easy problem) → std = 0, advantage = 0
        • All N_SAMPLES responses wrong    (hard problem) → std = 0, advantage = 0

    Args:
        responses: list of response dicts (length = n_groups * n_samples_per_prompt)
        rewards:   list of float, same length as responses
        n_samples_per_prompt: int

    Returns:
        (filtered_responses, filtered_rewards, n_filtered)
            filtered_* have all zero-std groups removed (still grouped in multiples of
            n_samples_per_prompt, so GRPO reshaping in the training server stays valid)
            n_filtered is the number of groups that were dropped
    """
    assert len(responses) % n_samples_per_prompt == 0, (
        f"len(responses)={len(responses)} not divisible by "
        f"n_samples_per_prompt={n_samples_per_prompt}"
    )
    n_groups = len(responses) // n_samples_per_prompt

    kept_responses = []
    kept_rewards = []
    n_filtered = 0

    for g in range(n_groups):
        start = g * n_samples_per_prompt
        end = start + n_samples_per_prompt
        group_rewards = rewards[start:end]

        # std > 0 iff not all rewards in the group are the same value
        if len(set(group_rewards)) > 1:
            kept_responses.extend(responses[start:end])
            kept_rewards.extend(group_rewards)
        else:
            n_filtered += 1

    return kept_responses, kept_rewards, n_filtered


# --- Test the filter on the batch we already have ---
filtered_responses, filtered_rewards, n_filtered = filter_zero_std_groups(
    responses, rewards, N_SAMPLES_PER_PROMPT
)
n_groups = len(responses) // N_SAMPLES_PER_PROMPT
n_kept = len(filtered_responses) // N_SAMPLES_PER_PROMPT
print(f"Groups: {n_groups} total → {n_kept} kept, {n_filtered} filtered (zero reward std)")
if filtered_rewards:
    print(f"Mean reward (kept groups): {sum(filtered_rewards) / len(filtered_rewards):.3f}")

def package_rollout_data(responses, rewards, n_samples_per_prompt):
    """Package rollout results for the training server.

    Creates the data format expected by fsdp_training_server.py /train_step.
    """
    return {
        "tokens": [r["full_token_ids"] for r in responses],
        "rollout_log_probs": [r["rollout_log_probs"] for r in responses],
        "rewards": rewards,
        "prompt_lengths": [r["prompt_length"] for r in responses],
        "response_lengths": [r["response_length"] for r in responses],
        "n_samples_per_prompt": n_samples_per_prompt,
    }

# Memory transition: Rollout → Train (same as training loop step 5)
requests.post(f"{SGLANG_URL}/release_memory_occupation", json={}, timeout=60)
requests.post(f"{TRAIN_URL}/wake_up", json={}, timeout=120)

# Send to training server
rollout_data = package_rollout_data(responses, rewards, N_SAMPLES_PER_PROMPT)

train_result = requests.post(f"{TRAIN_URL}/train_step", json=rollout_data, timeout=300)
train_metrics = train_result.json()

print("Training step result:")
for k, v in train_metrics.items():
    print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

model_info = requests.get(f"{SGLANG_URL}/model_info").json()
print(f"Before update: SGLang weight version: {model_info.get('weight_version', 'N/A')}")

# Memory transition: Train → Weight sync (same as training loop step 7)
# Sleep FSDP first, then resume SGLang weights for the copy target
requests.post(f"{TRAIN_URL}/sleep", json={}, timeout=120)
requests.post(f"{SGLANG_URL}/resume_memory_occupation", json={"tags": ["weights"]}, timeout=60)

# Sync weights from training engine to inference engine
update_result = requests.post(
    f"{TRAIN_URL}/update_weights",
    json={"sglang_url": SGLANG_URL, "tp_size": TP_SIZE},
    timeout=300,
)
print(f"Weight update result: {update_result.json()}")

# Memory transition: Weight sync → Rollout (same as training loop step 8)
requests.post(f"{SGLANG_URL}/resume_memory_occupation", json={"tags": ["cuda_graph", "kv_cache"]}, timeout=60)

# Verify weight version on SGLang
model_info = requests.get(f"{SGLANG_URL}/model_info").json()
print(f"After update: SGLang weight version: {model_info.get('weight_version', 'N/A')}")


EVAL_SIZE = 1319  # Number of test problems to evaluate on

async def evaluate(sglang_url, test_data, tokenizer, num_samples=EVAL_SIZE):
    """Evaluate model accuracy on the GSM8K test set.

    Uses greedy decoding (temperature=0) for deterministic results.
    Mirrors the evaluation loop from miles/rollout/sglang_rollout.py.

    Returns:
        (accuracy, n_correct, n_total)
    """
    eval_subset = test_data[:num_samples]

    async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
        tasks = []
        labels = []

        for item in eval_subset:
            formatted = format_prompt(item["question"], tokenizer)
            payload = {
                "input_ids": formatted["token_ids"],
                "sampling_params": {
                    "temperature": 0,          # greedy — deterministic
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "skip_special_tokens": True,
                },
            }
            tasks.append(client.post(f"{sglang_url}/generate", json=payload))
            labels.append(item["label"])

        responses = await asyncio.gather(*tasks)

    correct = sum(
        grade_answer_verl(resp.json().get("text", ""), label)
        for resp, label in zip(responses, labels)
    )
    accuracy = correct / len(labels)
    return accuracy, correct, len(labels)


# ── Baseline evaluation ────────────────────────────────────────────────────
# Run BEFORE any training so we have a reference point to compare against.
# We reuse the same evaluate() function for the post-training measurement in
# Section 10, ensuring the comparison is apples-to-apples.
print(f"Baseline evaluation on {EVAL_SIZE} GSM8K test problems (greedy decoding)...")

# initial_accuracy, initial_correct, initial_total = await evaluate(SGLANG_URL, test_data, tokenizer)
initial_accuracy, initial_correct, initial_total = asyncio.run(evaluate(SGLANG_URL, test_data, tokenizer))
print(f"Baseline accuracy: {initial_correct}/{initial_total} = {initial_accuracy*100:.1f}%")

def print_gpu_memory():
    """Print GPU memory usage for all GPUs via nvidia-smi."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    for line in result.stdout.strip().split("\n"):
        idx, used, total, free = [x.strip() for x in line.split(",")]
        print(f"  GPU {idx}: {used}/{total} MiB used ({free} MiB free)")

# Reset data loader for training
data_loader = GSM8KDataLoader(
    train_data,
    tokenizer,
    batch_size=BATCH_SIZE,
    n_samples_per_prompt=N_SAMPLES_PER_PROMPT,
)

metrics_history = []

for iteration in range(NUM_ITERATIONS):
    iter_start = time.time()

    # 1. Sample batch
    batch = data_loader.get_batch()

    # 2. Rollout (SGLang active on GPU, FSDP sleeping on CPU)
    
    # responses = await rollout_batch(SGLANG_URL, batch, ROLLOUT_TEMPERATURE, MAX_NEW_TOKENS)
    responses = asyncio.run(rollout_batch(SGLANG_URL, batch, ROLLOUT_TEMPERATURE, MAX_NEW_TOKENS))
    rollout_time = time.time() - iter_start

    # 3. Compute rewards
    rewards = [
        1.0 if grade_answer_verl(r["response_text"], r["label"]) else 0.0
        for r in responses
    ]
    mean_reward = sum(rewards) / len(rewards)

    # 4. Filter groups with zero reward std (no GRPO signal)
    responses, rewards, n_filtered = filter_zero_std_groups(responses, rewards, N_SAMPLES_PER_PROMPT)
    if not responses:
        print(f"[Iter {iteration:3d}] All {BATCH_SIZE} groups filtered (zero reward std) — skipping")
        continue

    # 5. Memory transition: Rollout → Train
    #    Mirrors train.py:73-79: offload ALL SGLang memory, then wake up FSDP.
    requests.post(f"{SGLANG_URL}/release_memory_occupation", json={}, timeout=60)
    requests.post(f"{TRAIN_URL}/wake_up", json={}, timeout=120)

    # 6. Train
    rollout_data = package_rollout_data(responses, rewards, N_SAMPLES_PER_PROMPT)
    train_result = requests.post(
        f"{TRAIN_URL}/train_step", json=rollout_data, timeout=300
    ).json()
    train_time = time.time() - iter_start - rollout_time

    # 7. Memory transition: Train → Weight sync
    #    Mirrors train.py:92-95:
    #      offload_train()                          → sleep FSDP to CPU
    #      rollout_manager.onload_weights()          → resume SGLang weights only
    #      actor_model.update_weights()              → push weights (per-bucket .cuda())
    requests.post(f"{TRAIN_URL}/sleep", json={}, timeout=120)
    requests.post(f"{SGLANG_URL}/resume_memory_occupation", json={"tags": ["weights"]}, timeout=60)
    requests.post(
        f"{TRAIN_URL}/update_weights",
        json={"sglang_url": SGLANG_URL, "tp_size": TP_SIZE},
        timeout=300,
    )

    # 8. Memory transition: Weight sync → Rollout
    #    Mirrors train.py:96-97: rollout_manager.onload_kv()
    requests.post(f"{SGLANG_URL}/resume_memory_occupation", json={"tags": ["cuda_graph", "kv_cache"]}, timeout=60)
    total_time = time.time() - iter_start

    # 9. Log metrics
    metrics = {
        "iteration": iteration,
        "mean_reward": mean_reward,
        "n_filtered": n_filtered,
        "loss": train_result["loss"],
        "grad_norm": train_result["grad_norm"],
        "clipfrac": train_result["clipfrac"],
        "rollout_time": rollout_time,
        "train_time": train_time,
        "total_time": total_time,
    }
    metrics_history.append(metrics)

    print(
        f"[Iter {iteration:3d}] "
        f"reward={mean_reward:.3f}  "
        f"filtered={n_filtered}/{BATCH_SIZE}  "
        f"loss={train_result['loss']:.4f}  "
        f"grad_norm={train_result['grad_norm']:.4f}  "
        f"clip={train_result['clipfrac']:.3f}  "
        f"time={total_time:.1f}s (rollout={rollout_time:.1f}s train={train_time:.1f}s)"
    )

    # Print GPU memory every 10 iterations
    if (iteration + 1) % 10 == 0:
        print(f"--- GPU Memory at iteration {iteration} (rollout phase) ---")
        print_gpu_memory()

print("\nTraining complete!")
print("--- Final GPU Memory ---")
print_gpu_memory()




# Plot training curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

iters = [m["iteration"] for m in metrics_history]

axes[0].plot(iters, [m["mean_reward"] for m in metrics_history], "b-o")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Mean Reward")
axes[0].set_title("Training Reward")
axes[0].grid(True)

axes[1].plot(iters, [m["loss"] for m in metrics_history], "r-o")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Loss")
axes[1].set_title("Policy Loss")
axes[1].grid(True)

axes[2].plot(iters, [m["grad_norm"] for m in metrics_history], "g-o")
axes[2].set_xlabel("Iteration")
axes[2].set_ylabel("Gradient Norm")
axes[2].set_title("Gradient Norm")
axes[2].grid(True)

plt.tight_layout()
plt.show()



# --- Final evaluation (after training) ---
print(f"Evaluating trained model on {EVAL_SIZE} GSM8K test problems...")

# final_accuracy, final_correct, final_total = await evaluate(SGLANG_URL, test_data, tokenizer)
final_accuracy, final_correct, final_total = asyncio.run(evaluate(SGLANG_URL, test_data, tokenizer))

delta_pp = (final_accuracy - initial_accuracy) * 100
print(f"\n{'='*45}")
print(f"  GSM8K Accuracy Results ({final_total} problems)")
print(f"{'='*45}")
print(f"  Before training : {initial_correct:3d}/{initial_total} = {initial_accuracy*100:5.1f}%")
print(f"  After  training : {final_correct:3d}/{final_total} = {final_accuracy*100:5.1f}%")
print(f"  Improvement     : {delta_pp:+.1f} pp")
print(f"{'='*45}")

# cleanup
sglang_proc.terminate()
train_proc.terminate()
print("Servers terminated.")
