"""
train.py
--------
Training loop for the Diff_MRTA multi-UAV diffusion framework.
Loads multiple JSON scenario files from a specified folder, trains the 
ThreeHeadedDenoiser to reverse the diffusion process, and manages checkpoints.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Import custom project modules
from model import ThreeHeadedDenoiser, make_beta_schedule
from scenario_utils import ScenarioSampler, build_robot_features, build_task_features, build_balanced_random_schedule
# If using RL later, you will import: from objectives_and_reward import compute_objectives, compute_rewards

# =============================================================================
# 1. HYPERPARAMETERS & CONFIGURATION
# =============================================================================

# --- Data & Directory Settings ---
# Folder containing all your scenario_{R}_{T}_{seed}.json files. 
# The sampler will automatically discover and load all JSONs inside.
SCENARIO_FOLDER = "./scenarios"  
# Directory where model weights will be saved during training.
CHECKPOINT_DIR = "./checkpoints" 
# To resume training, set this to a file path (e.g., "./checkpoints/model_epoch_50.pt"). Set to None to start fresh.
RESUME_CHECKPOINT = None         

# --- Training Loop Parameters ---
# Total number of complete passes through the training phase.
EPOCHS = 2000                    
# How many batches to process per epoch (useful if sampling randomly from a massive folder of JSONs).
STEPS_PER_EPOCH = 100            
# How many scenarios to process simultaneously. (Note: Currently set to 1 as decoding handles unbatched logic, can scale up with vectorized decode).
BATCH_SIZE = 1                   
# The step size for the AdamW optimizer. Controls how fast the model updates its weights.
LEARNING_RATE = 1e-4             
# L2 Regularization term to prevent the model from overfitting to specific JSONs.
WEIGHT_DECAY = 1e-5              
# How often (in epochs) to save a model checkpoint to disk.
SAVE_INTERVAL = 100               

# --- Model Architecture Parameters ---
# The hidden dimension size for the Transformer and Embeddings.
D_MODEL = 128                     
# Number of attention heads in the Transformer encoder.
N_HEAD = 8                       
# Number of sequential Transformer encoder layers.
NUM_LAYERS = 3                   
# Dimension of user preference vector (e.g., [makespan, variance, energy]).
PREF_DIM = 3                     
# Minimum allowed travel velocity (meters/second) for the Beta distribution head.
MIN_VEL = 1.0                    
# Maximum allowed travel velocity (meters/second) for the Beta distribution head.
MAX_VEL = 15.0                   
# The absolute maximum number of robots the model architecture can embed (must be >= max R in your JSONs).
MAX_R = 20                       
# The absolute maximum number of tasks the model architecture can embed (must be >= max T in your JSONs).
MAX_T = 60                       

# --- Diffusion Parameters ---
# How many steps the Markov chain takes to go from pure noise to a clean schedule.
DIFFUSION_STEPS = 100            
# The noise level at the very first forward diffusion step.
BETA_START = 0.001               
# The noise level at the final forward diffusion step.
BETA_END = 0.2                   

# --- Hardware ---
# Automatically utilize a GPU if one is installed and configured, otherwise fallback to CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================


def save_checkpoint(epoch, model, optimizer, loss, filename):
    """Saves the model weights, optimizer states, and epoch to disk."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)
    print(f"[Info] Checkpoint saved: {filepath}")


def load_checkpoint(filepath, model, optimizer):
    """Loads a saved checkpoint to resume training exactly where it left off."""
    if not os.path.exists(filepath):
        print(f"[Warning] Checkpoint {filepath} not found. Starting from scratch.")
        return 0, None
    
    print(f"[Info] Loading checkpoint from {filepath}...")
    checkpoint = torch.load(filepath, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"[Info] Resuming from Epoch {epoch} (Previous Loss: {loss:.4f})")
    return epoch, loss


def train():
    # ---------------------------------------------------------
    # 1. Dataset Initialization
    # ---------------------------------------------------------
    print(f"Initializing Scenario Sampler from folder: {SCENARIO_FOLDER}...")
    # This automatically scans the folder and prepares all JSON files for training.
    sampler = ScenarioSampler(scenario_dir=SCENARIO_FOLDER, seed=42)
    
    # Peek at one scenario to dynamically extract feature dimensions
    dummy_scenario, _, _ = sampler.sample()
    r_feats_dummy = build_robot_features(dummy_scenario)
    t_feats_dummy = build_task_features(dummy_scenario)
    
    robot_feat_dim = r_feats_dummy.shape[-1]
    task_feat_dim = t_feats_dummy.shape[-1]

    # ---------------------------------------------------------
    # 2. Model & Optimizer Initialization
    # ---------------------------------------------------------
    model = ThreeHeadedDenoiser(
        robot_feat_dim=robot_feat_dim, 
        task_feat_dim=task_feat_dim,
        d_model=D_MODEL, 
        nhead=N_HEAD, 
        num_layers=NUM_LAYERS, 
        pref_dim=PREF_DIM,
        min_vel=MIN_VEL, 
        max_vel=MAX_VEL,
        max_R=MAX_R, 
        max_T=MAX_T
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # ---------------------------------------------------------
    # 3. Resume Checkpoint Logic
    # ---------------------------------------------------------
    start_epoch = 1
    if RESUME_CHECKPOINT:
        start_epoch, _ = load_checkpoint(RESUME_CHECKPOINT, model, optimizer)
        start_epoch += 1  # Start at the next epoch following the saved one

    # ---------------------------------------------------------
    # 4. Main Training Loop
    # ---------------------------------------------------------
    print(f"\nStarting Training on {DEVICE.upper()}...")
    model.train()
    
    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_loss = 0.0

        for step in range(STEPS_PER_EPOCH):
            optimizer.zero_grad()
            
            # --- A. Data Loading ---
            # Sample a random JSON scenario from the folder
            scenario, filepath, _ = sampler.sample()
            R, T = scenario["R"], scenario["T"]
            
            r_feats = torch.tensor(build_robot_features(scenario), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            t_feats = torch.tensor(build_task_features(scenario), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            # Simulate a random user preference (e.g., weighting makespan vs energy)
            pref = torch.rand(1, PREF_DIM, dtype=torch.float32).to(DEVICE)

            # --- B. Generate Target (Pre-training / Behavior Cloning) ---
            # For pure RL, you would use model outputs to interact with objectives_and_reward.py.
            # Here, we generate a feasible balanced schedule to train the denoiser to reconstruct valid shapes.
            target_schedule_np = build_balanced_random_schedule(scenario)
            
            # Extract target assignments: Which robot gets which task (1D array of length T)
            # Find the row index (robot) where the task column is > 0
            target_assignments = np.zeros(T, dtype=np.int64)
            for t_idx in range(T):
                assigned_r = np.where(target_schedule_np[:, t_idx] > 0)[0]
                # If assigned, use the robot index; otherwise set it to R (unassigned marker)
                target_assignments[t_idx] = assigned_r[0] if len(assigned_r) > 0 else R
                
            target_assignments_t = torch.tensor(target_assignments, dtype=torch.long).to(DEVICE)
            
            # --- C. Diffusion Step Simulation ---
            # Sample a random diffusion timestep (t)
            t = torch.randint(1, DIFFUSION_STEPS + 1, (1,)).to(DEVICE)
            
            # Simulate a noisy categorical state (x_t) by corrupting the target schedule
            # (In a full D3PM, this is done via the cumulative transition matrix Q_bar)
            noise_mask = torch.rand(1, T).to(DEVICE) < (t.float() / DIFFUSION_STEPS)
            random_assignments = torch.randint(0, R + 1, (1, T)).to(DEVICE)
            x_t_noisy = torch.where(noise_mask, random_assignments, target_assignments_t.unsqueeze(0))

            # --- D. Forward Pass ---
            assign_logits, rank_raw, velocity, alpha, beta = model(
                r_feats, t_feats, pref, x_t=x_t_noisy, t=t
            )

            # --- E. Loss Computation ---
            # 1. Assignment Loss: CrossEntropy between predicted logits and target assignments
            # If target_assignments_t contains 'R' (meaning unassigned), we use ignore_index=R 
            # or ensure target values strictly stay within [0, R-1].
            # Here we cap/ignore out-of-bound target values gracefully:
            valid_mask = (target_assignments_t >= 0) & (target_assignments_t < R)

            if valid_mask.sum() > 0:
                # Option A: Use ignore_index for unassigned tasks (if unassigned is marked as R)
                loss_assign = F.cross_entropy(
                    assign_logits.squeeze(0).transpose(0, 1), 
                    target_assignments_t, 
                    ignore_index=R  # Tells CrossEntropy to skip targets equal to R
                )
            else:
                loss_assign = torch.tensor(0.0, device=assign_logits.device, requires_grad=True)
            
            # 2. Rank/Velocity Loss (Mocked for unassigned structural training)
            # In RL mode, this is replaced by: -Reward * log_prob(action)
            loss_rank = rank_raw.mean() * 0.0  # Placeholder 
            loss_vel = alpha.mean() * 0.0      # Placeholder
            
            total_loss = loss_assign + loss_rank + loss_vel

            # --- F. Backpropagation ---
            total_loss.backward()
            
            # Prevent exploding gradients (common in Transformer encoders)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()

            epoch_loss += total_loss.item()

        # --- Epoch Summary & Checkpointing ---
        avg_epoch_loss = epoch_loss / STEPS_PER_EPOCH
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{EPOCHS}] | Avg Loss: {avg_epoch_loss:.4f}")
        # print(f"Epoch [{epoch}/{EPOCHS}] | Avg Loss: {avg_epoch_loss:.4f}")

        if epoch % SAVE_INTERVAL == 0:
            filename = f"model_epoch_{epoch}.pt"
            save_checkpoint(epoch, model, optimizer, avg_epoch_loss, filename)

    print("Training Complete!")


if __name__ == "__main__":
    train()