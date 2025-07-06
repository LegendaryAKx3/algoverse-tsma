"""
Train a Sparse Autoencoder (SAE) on GPT-2 residuals using SAE-Lens.
"""

import torch
import wandb
from transformer_lens import HookedTransformer
from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner

def main():
    # Initialize wandb with proper login handling
    try:
        # Try to login to wandb (will prompt for API key if not logged in)
        wandb.login()
        
        # Initialize wandb
        wandb.init(
            project="sparse-autoencoder-gpt2",
            name="gpt2-sae-layer8",
            config={
                "model": "gpt2",
                "hook_layer": 8,
                "expansion_factor": 8,
                "l1_coefficient": 1e-3,
                "learning_rate": 1e-4,
                "training_tokens": 100_000,
            },
            mode="online"  # Set to "offline" if you want to run without internet
        )
        print("Wandb intialized")
    except Exception as e:
        print(f"Wandb initialization failed: {e}")
        # Initialize wandb in offline mode as fallback
        wandb.init(mode="disabled")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load & hook GPT-2 via Transformer-Lens
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    # Choose a residual stream hook point (e.g. before block 9)
    hook_name = "blocks.8.hook_resid_pre"
    model.reset_hooks()

    # 3. Configure SAE-Lens training
    cfg = LanguageModelSAERunnerConfig(
        # Basic required parameters
        model_name="gpt2",
        hook_name=hook_name,
        hook_layer=8,
        d_in=768,  # GPT-2 base model dimension
        expansion_factor=8,
        
        # Training parameters
        lr=1e-4,
        l1_coefficient=1e-3,
        training_tokens=100_000,  # Reduced for testing
        
        # Use a simple dataset approach
        dataset_path="NeelNanda/pile-10k",  # Smaller, more reliable dataset
        streaming=True,
        
        # Wandb integration
        log_to_wandb=True,
        wandb_project="sparse-autoencoder-gpt2",
        
        device=device,
    )

    # 4. Instantiate and run training
    runner = SAETrainingRunner(cfg)
    print("Starting SAE training...")
    sae = runner.run()
    print("Training complete.")

    
    # Finish wandb run
    wandb.finish()

main()
