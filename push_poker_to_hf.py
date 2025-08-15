# modal_push_hf.py
# Push an existing checkpoint (already saved in a Modal volume) to the Hugging Face Hub.

import os
import modal

# Silence tokenizers' fork warning in case your env imports it elsewhere
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = modal.App("push-checkpoint-to-hf")

# Your persisted volume that already contains the trained checkpoint
VOL_NAME = "tsma"  # <-- adjust only if different
vol = modal.Volume.from_name(VOL_NAME)

# Minimal image with the HF Hub client
image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub>=0.23.0")
)

@app.function(
    image=image,
    volumes={"/data": vol},                       # mount the volume at /data
    cpu=2, memory=4 * 1024,
    timeout=60 * 30,                              # 30 minutes should be plenty
    secrets=[modal.Secret.from_name("huggingface-secret")]  # should provide HF_TOKEN
)
def push_to_hf(
    repo_id: str,
    path_in_volume: str = "runs/run1",   # e.g. where Trainer.save_model() wrote files
    private: bool = True,
    commit_message: str = "Add checkpoint",
    repo_type: str = "model",            # "model" or "dataset" or "space"
    create_pr: bool = False,
):
    """
    Push a folder from the Modal volume to the Hugging Face Hub.

    Args:
      repo_id: e.g. "username/poker-gpt-scratch"
      path_in_volume: path inside the mounted volume (/data) to upload
      private: create repo as private if it doesn't exist
      commit_message: commit message for the upload
      repo_type: Hub repo type ("model" most likely)
      create_pr: if True, upload as a PR instead of pushing to main
    """
    from huggingface_hub import HfApi, create_repo, upload_folder

    # Expect HF token from secret (configure a Modal secret named "huggingface"
    # that includes env var HF_TOKEN=<your token>)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF token not found. Add a Modal secret with HF_TOKEN, "
            "and include it in the function via secrets=[modal.Secret.from_name('huggingface')]."
        )

    api = HfApi(token=hf_token)

    # Ensure repo exists (no-op if it already does)
    create_repo(repo_id, exist_ok=True, private=private, repo_type=repo_type)

    src = os.path.join("/data", path_in_volume)
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"Path '{src}' not found in volume '{VOL_NAME}'. "
            f"List volume contents with: `modal volume ls {VOL_NAME}`"
        )

    print(f"Uploading from: {src}")
    print(f"Destination repo: {repo_id} (private={private}, type={repo_type})")

    # Optional: ignore large temp/log files. Adjust as needed.
    ignore_patterns = [
        "**/logs/**",
        "**/events.out.tfevents*",
        "**/*.tmp",
        "**/__pycache__/**",
        "**/.DS_Store",
        "**/.git/**",
    ]

    # Upload folder (idempotent; will create/update files on the Hub)
    upload_folder(
        folder_path=src,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message,
        ignore_patterns=ignore_patterns,
        create_pr=create_pr,
    )

    # Print the commit URL for convenience
    info = api.repo_info(repo_id, repo_type=repo_type)
    web_url = f"https://huggingface.co/{repo_id}"
    print(f"✅ Upload complete. Repo page: {web_url}")
    if create_pr:
        print("Note: You uploaded as a Pull Request; open the repo page to review/merge.")

@app.local_entrypoint()
def main(
    repo_id: str = "krangana/poker-gpt-scratch",
    path_in_volume: str = "runs/run1",
    private: bool = True,
    create_pr: bool = False,
):
    # Forward to the remote function
    push_to_hf.remote(
        repo_id=repo_id,
        path_in_volume=path_in_volume,
        private=private,
        create_pr=create_pr,
    )

