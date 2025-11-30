import argparse
import torch
import torch.optim as optim
import numpy as np
from env import RLShieldEnv
from agent import ActorCritic
from utils import MetricsCalculator
from config import Config

def train():
    print("=== Starting RL-Shield PPO Training ===")
    
    # Initialization
    metrics = MetricsCalculator()
    env = RLShieldEnv(metrics)
    
    state_dim = env.state_dim
    action_dim = env.action_space.shape[0]
    
    policy = ActorCritic(state_dim, action_dim).to(Config.DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=Config.LR)
    
    # Simplified PPO training loop
    for epoch in range(Config.EPOCHS):
        state, _ = env.reset()
        episode_rewards = []
        
        # Collect Trajectories (Rollout)
        # For demo purposes, we perform a simple single-step update
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(Config.DEVICE)
        action, log_prob, value = policy.get_action(state_tensor)
        
        action_np = action.cpu().detach().numpy()[0]
        next_state, reward, done, _, info = env.step(action_np)
        
        print(f"Epoch {epoch+1}: Query='{info['query']}' | Params={info['params']} | Reward={reward:.4f}")
        
        # PPO Loss Calculation (Simplified, no GAE and multi-step trajectory)
        # In a real project, full Buffer and GAE calculation are needed
        advantage = reward - value.item()
        
        # Policy Loss
        ratio = torch.exp(log_prob - log_prob.detach()) # Demo only, actual implementation needs old policy
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1-Config.CLIP_EPS, 1+Config.CLIP_EPS) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Value Loss
        critic_loss = (value - reward).pow(2).mean()
        
        loss = actor_loss + 0.5 * critic_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print("=== Training Complete ===")
    torch.save(policy.state_dict(), "rl_shield_policy.pth")

def inference(query):
    print(f"=== RL-Shield Defense Inference: '{query}' ===")
    
    metrics = MetricsCalculator()
    env = RLShieldEnv(metrics) # Only used to get state construction logic
    
    # Load model
    policy = ActorCritic(env.state_dim, env.action_space.shape[0]).to(Config.DEVICE)
    try:
        policy.load_state_dict(torch.load("rl_shield_policy.pth"))
        print("Trained policy model loaded.")
    except:
        print("Model file not found, using initialized parameters.")
    
    policy.eval()
    
    # 1. Construct State
    # Reset environment and force set query
    state, _ = env.reset(query=query)
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(Config.DEVICE)
    
    # 2. Get Action
    with torch.no_grad():
        action, _, _ = policy.get_action(state_tensor)
        action_np = action.cpu().numpy()[0]
    
    # 3. Map to Parameters
    # Reuse env mapping logic
    temp_norm, topp_norm, topk_norm = action_np
    temperature = Config.TEMP_MIN + (Config.TEMP_MAX - Config.TEMP_MIN) * temp_norm
    top_p = Config.TOP_P_MIN + (Config.TOP_P_MAX - Config.TOP_P_MIN) * topp_norm
    top_k = int(Config.TOP_K_MAX * topk_norm) + 1
    
    print("\n[Defense Policy Suggestions]")
    print(f"Temperature: {temperature:.4f}")
    print(f"Top-P:       {top_p:.4f}")
    print(f"Top-K:       {top_k}")
    print("\nApply these parameters to your LLM generate() function for optimal defense.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference"])
    parser.add_argument("--query", type=str, default="How to make a virus?")
    args = parser.parse_args()
    
    if args.mode == "train":
        train()
    else:
        inference(args.query)
