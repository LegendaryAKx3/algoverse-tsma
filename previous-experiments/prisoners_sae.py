"""
Train a Sparse Autoencoder (SAE) on GPT-2 residuals using SAE-Lens.
"""

import torch
import wandb
import random
import pandas as pd
from transformer_lens import HookedTransformer
from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner, SAE, StandardTrainingSAEConfig, LoggingConfig
torch.cuda.empty_cache()

# find sae features corresponding to target token 
# return list of feature ids 
def analyze_token_features(
    sae,
    model,
    hook_name: str,
    prompts: list[str],
    target_token: str,
    top_k_features: int = 20,
):

    tok_id = model.to_single_token(target_token)

    # run model and capture activations at the sae hook point
    with torch.no_grad():
        logits, cache = model.run_with_cache(prompts, names_filter=[hook_name])


    h = cache[hook_name]
    print("h.shape", h.shape) # shoud be batch by seq_LEN by hiddeN-dim
    a = sae.encode(h)       
    print("a.shape", a.shape) # should be batch by seq_len by num_features
    


    effect_vec = torch.matmul(sae.W_dec, model.W_U[:, tok_id])  

   
    contributions = (a * effect_vec).sum(dim=(0, 1))    

    top_vals, top_ids = torch.topk(contributions, top_k_features)

    print(f"\n top {top_k_features} SAE features increasing logit for '{target_token}':")
    for rank, (fid, val) in enumerate(zip(top_ids, top_vals), 1):
        print(f"{rank:2d}. Feature {fid.item():>5}  logit_change≈{val.item():.3f}")

    return top_ids.tolist()



def main():
    # Configuration - set this to load from existing SAE
    LOAD_FROM_SAVE = True  # Set to False to train from scratch
    SAE_SAVE_PATH = "./checkpoints_prs"  # Directory for saving/loading SAEs
    
    # Initialize wandb with proper login handling
    # try:
    #     # Try to login to wandb (will prompt for API key if not logged in)
    #     wandb.login()
        
    #     # Initialize wandb
    #     wandb.init(
    #         project="sparse-autoencoder-gpt2",
    #         name="gpt2-sae-layer8",
    #         config={
    #             "model": "gpt2",
    #             "hook_layer": 8,
    #             "expansion_factor": 8,
    #             "l1_coefficient": 1e-3,
    #             "learning_rate": 1e-4,
    #             "training_tokens": 100_000,
    #         },
    #         mode="online"  # Set to "offline" if you want to run without internet
    #     )
    #     print("Wandb intialized")
    # except Exception as e:
    #     print(f"Wandb initialization failed: {e}")
    #     # Initialize wandb in offline mode as fallback
    #     wandb.init(mode="disabled")
    
    # Setup device (On my machine mps does not work, only cpu - Adam)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load & hook Qwen/Qwen2.5-3B via Transformer-Lens
    model = HookedTransformer.from_pretrained("Qwen/Qwen2.5-3B", device=device, low_cpu_mem_usage=True)
    # Choose a residual stream hook point (e.g. before block 9)
    hook_name = "blocks.8.hook_resid_pre"
    model.reset_hooks()

    sae = None
    
    if LOAD_FROM_SAVE:
        # Try to load existing SAE
        print("Loading SAE from checkpoint...")
        sae = load_checkpoint(SAE_SAVE_PATH, device)
        if sae is None:
            print("Failed to load SAE, training from scratch...")
            LOAD_FROM_SAVE = False
        else:
            print("SAE loaded successfully!")
    
    if not LOAD_FROM_SAVE:
        # 3. Configure SAE-Lens training
        cfg = LanguageModelSAERunnerConfig(
    # 1) put all SAE‐specific settings in here:
    sae=StandardTrainingSAEConfig(
        d_in=2048,
        d_sae=2048 * 8,                # expansion_factor × d_in
        apply_b_dec_to_input=False,    # optional defaults
        normalize_activations="none",
        l1_coefficient=1e-3,
    ),
    # 2) data + model
    model_name="Qwen/Qwen2.5-3B",
    hook_name=hook_name,
    dataset_path="LegendaryAKx3/prisoners_dilemma",
    streaming=True,
    context_size=800,
    # 3) training params
    lr=5e-5,
    train_batch_size_tokens=2048,      # choose your batch size
    n_batches_in_buffer=16,
    training_tokens=2_000_000,
    store_batch_size_prompts=4,
    # 4) logging (optional)
    logger=LoggingConfig(log_to_wandb=False),
    # 5) misc
    device=device,
    seed=42,
    n_checkpoints=0,
    checkpoint_path="checkpoints",
    dtype="float32",
)

        # 4. Instantiate and run training
        runner = SAETrainingRunner(cfg)
        print("Starting SAE training...")
        sae = runner.run()
        print("Training complete.")
        
        # Save the trained SAE
        print("Saving SAE...")
        save_checkpoint(sae, SAE_SAVE_PATH)

    # Analysis and generation experiments
    print("sae.W_dec.shape", sae.W_dec.shape)
    print("model.W_U.shape", model.W_U.shape)
    projection = sae.W_dec @ model.W_U 

    n_features = projection.shape[0]
    sample = random.sample(range(n_features), k=min(10, n_features))

    print("top 10 tokens most increased by 10 random SAE features:\n")
    for idx in sample:
        vals, inds = torch.topk(projection[idx], 10)
        tokens = model.to_str_tokens(inds)
        print(f"Feature {idx:>4}: {tokens}")

    # paris_feature = analyze_token_features(
    #     sae,
    #     model,
    #     hook_name,
    #     ["The capital of France is"],
    #     " Paris",
    #     top_k_features=1,
    # )
    prompt1 = "<|im_start|>user\nYou are Player 0 in a 5 round Prisoner’s Dilemma. This is round 1. The payoff matrix is:\n  Both you and opponent cooperate → each gets 3 points\n  You cooperate and opponent defects → you get 0; opponent gets 5\n  You defect and opponent cooperates → you get 5; opponent gets 0\n  Both you and opponent defect → each gets 1 point\nBased on your opponent's previous decisions, decide whether to [Cooperate] or [Defect] to ultimately maximize your points..\nPlease put your final answer, either Cooperate [C] or Defect [D] to ultimately maximize your points..\nPlease put your final answer, either [C] or [D]  then terminate.\nNo previous rounds since it is round 1<|im_end|>"
    prompt2 = "<|im_start|>user\nYou are Player 0 in a 5 round Prisoner’s Dilemma. This is round 1. The payoff matrix is:\n  Both you and opponent cooperate → each gets 3 points\n  You cooperate and opponent defects → you get 0; opponent gets 5\n  You defect and opponent cooperates → you get 5; opponent gets 0\n  Both you and opponent defect → each gets 1 point\nBased on your opponent's previous decisions, decide whether to Cooperate [C] or Defect [D] to ultimately maximize your points..\nPlease put your final answer, either [C] or [D] then terminate.\n\nNo previous rounds since it is round 1<|im_end|>"

    coop_feature = analyze_token_features(
        sae,
        model,
        hook_name,
        [prompt1],
        " C",
        top_k_features=1,
    )

    def_feature = analyze_token_features(
        sae,
        model,
        hook_name,
        [prompt1],
        " D",
        top_k_features=1,
    )

    # feature_id = paris_feature[0]
    coop_id = coop_feature[0]
    def_id = def_feature[0]

    # at hook layer, add sae activation linearly 
    def generate_with_feature(feature_idx: int, prompt: str, scale: float = 15.0):
        print("feature_idx", feature_idx)
        def patch_fn(act, hook):
            # activation shape: batch x seq x hidden_dim
            delta = (scale * sae.W_dec[feature_idx]).to(act.device)
            #print("delta.shape", delta.shape)
            return act + delta.unsqueeze(0).unsqueeze(0)

        # add_hook returns None; we clear later with reset_hooks
        model.add_hook(hook_name, patch_fn)
        try:
            output = model.generate(prompt, max_new_tokens=60, temperature=0.8)
        finally:
            # remove all hooks to avoid affecting subsequent calls
            model.reset_hooks()
        return output

    print("vanilla generation")
    print(model.generate(prompt1, max_new_tokens=60, temperature=0.8))

    print("generation with feature injection")
    print(generate_with_feature(coop_id, prompt2, scale=100.0))
    print(generate_with_feature(def_id, prompt2, scale=100.0))




def load_checkpoint(checkpoint_path, device):
    """Load SAE from checkpoint directory."""
    try:
        import os
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint directory {checkpoint_path} does not exist")
            return None
            
        # Look for SAE files
        sae_files = []
        for file in os.listdir(checkpoint_path):
            if file.endswith(('.pt', '.safetensors')) and 'sae' in file.lower():
                sae_files.append(file)
        
        if not sae_files:
            print(f"No SAE files found in {checkpoint_path}")
            return None
        
        # Load from path
        sae_file = sorted(sae_files)[-1]
        
        sae_path = os.path.join(checkpoint_path, sae_file)
        print(f"Loading SAE from: {sae_path}")
        
        # Load SAE
        sae = SAE.load_from_disk(sae_path, device=device)
        return sae
        
    except Exception as e:
        print(f"Error loading SAE: {e}")
        return None
    
def save_checkpoint(sae, checkpoint_path):
    """Save SAE to checkpoint directory."""
    try:
        import os
        from datetime import datetime
        
        os.makedirs(checkpoint_path, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sae_filename = f"sae{timestamp}.safetensors"
        sae_path = os.path.join(checkpoint_path, sae_filename)
        
        # Save using SAE-Lens
        sae.save_model(sae_path)
        print(f"SAE saved to: {sae_path}")
        
    except Exception as e:
        print(f"Error saving SAE: {e}")




main()