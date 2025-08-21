#!/usr/bin/env python3
import modal
import json, random, math, os, re, shutil
from typing import List

# ---------- App & Image ----------

app = modal.App("phh-six-max-gpu")

# Pick your GPU here: "A10G" (great), "L4" (great), "A100" (fastest), "T4" (budget)
GPU_KIND = "A100"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("treys")
    .pip_install("numpy")
    .pip_install("torch==2.5.1+cu121", index_url="https://download.pytorch.org/whl/cu121")
    # Bake your local `gen_torch.py` into the container so imports "just work"
    .add_local_file("gen_torch.py", remote_path="/root/gen_torch.py")
)

# Persistent storage for your datasets (survives across runs)
vol = modal.Volume.from_name("phh-dataset", create_if_missing=True)


# ---------- Core generator ----------

@app.function(image=image, gpu=GPU_KIND, volumes={"/vol": vol}, timeout=24 * 60 * 60)
def generate_poker_logs(
    hands: int = 100000,
    iters: int = 8192,
    stack_size: int = 10000,
    seed: int = 42,
    mc_chunk: int = 8192,
    autotune_chunk: bool = True,
    out_name: str = "dataset_part_00.ndjson",
    save_every: int = 100,
) -> str:
    """
    Generate poker logs on GPU and write NDJSON to /vol/<out_name>.
    """
    # Import here (image has gen_torch.py baked at /root/gen_torch.py)
    from gen_torch import simulate_hand, make_varied_styles, CudaEquity, autotune_mc_chunk

    rng = random.Random(seed)
    styles = make_varied_styles(seed)

    # CUDA equity engine
    cuda_engine = CudaEquity("cuda", mc_chunk=mc_chunk)
    try:
        _ = cuda_engine.equity_vs_n([0, 13], [], 1, 1, [])
        import torch

        torch.cuda.synchronize()
    except Exception:
        pass

    if autotune_chunk:
        tuned, last_ok = autotune_mc_chunk(cuda_engine, start=mc_chunk)
        cuda_engine.mc_chunk = tuned
        print(f"[autotune] mc_chunk tuned to {tuned} (last OK: {last_ok})")

    button = rng.randint(1, 6)
    buffer_lines: List[str] = []
    out_path = f"/vol/{out_name}"

    # Make sure parent dir exists inside the volume (if you pass nested out_name)
    os.makedirs(os.path.dirname(out_path) or "/vol", exist_ok=True)

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for h in range(hands):
            logs = simulate_hand(
                h, styles, button, iters, stack_size, rng, "cuda", cuda_engine
            )
            buffer_lines.append(json.dumps(logs) + "\n")
            button = (button % 6) + 1

            if len(buffer_lines) >= save_every:
                f.writelines(buffer_lines)
                f.flush()
                buffer_lines.clear()
                written += save_every
                vol.commit()  # persist progress
                print(f"[progress] wrote {written}/{hands} to {out_name}")

        if buffer_lines:
            f.writelines(buffer_lines)
            f.flush()

    vol.commit()
    msg = f"Saved {hands} hands to /vol/{out_name}"
    print(msg)
    return msg


# ---------- Utilities: list / preview / merge ----------

@app.function(image=image, volumes={"/vol": vol})
def list_volume() -> list[str]:
    return sorted([p for p in os.listdir("/vol")])


@app.function(image=image, volumes={"/vol": vol})
def head_file(path: str, n: int = 3) -> list[str]:
    full = f"/vol/{path}"
    out = []
    with open(full, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            out.append(line.rstrip("\n"))
    return out


@app.function(image=image, volumes={"/vol": vol})
def merge_files(prefix: str = "dataset_part_", merged: str = "dataset_500k.ndjson") -> str:
    parts = sorted(
        [p for p in os.listdir("/vol") if re.match(fr"{re.escape(prefix)}\d+\.ndjson$", p)]
    )
    if not parts:
        return "No part files found to merge."
    merged_path = f"/vol/{merged}"
    with open(merged_path, "wb") as out:
        for p in parts:
            with open(f"/vol/{p}", "rb") as f:
                shutil.copyfileobj(f, out)
    vol.commit()
    return f"Merged {len(parts)} parts into /vol/{merged}"


# ---------- Driver: shard to 500k and merge ----------

@app.local_entrypoint()
def run_big(
    total: int = 1000000,
    per_shard: int = 100000,   # 10 shards if you have a 10-GPU cap
    iters: int = 8192,
    stack_size: int = 10000,
    seed: int = 42,
    mc_chunk: int = 8192,
    autotune_chunk: bool = True,
    prefix: str = "dataset_part_",
    folder: str = "shards",    # where shards will be stored
):
    jobs = math.ceil(total / per_shard)
    print(f"[driver] Spawning {jobs} shards × {per_shard} = {total} hands")

    handles = []
    for i in range(jobs):
        out = f"{folder}/{prefix}{i:03d}.ndjson"
        h = generate_poker_logs.spawn(
            hands=per_shard,
            iters=iters,
            stack_size=stack_size,
            seed=seed + i,      # vary seed per shard
            mc_chunk=mc_chunk,
            autotune_chunk=autotune_chunk,
            out_name=out,
        )
        handles.append(h)

    for h in handles:
        print(h.get())