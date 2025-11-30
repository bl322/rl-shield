import argparse
import os
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter # 需要安装 tensorboard

from env import RLShieldEnv
from agent import ActorCritic
from utils import MetricsCalculator
from config import Config

def compute_gae(rewards, values, masks, next_value, gamma=0.99, tau=0.95):
    """完整版 GAE 计算，处理最后一步的 next_value"""
    gae = 0
    returns = []
    # 从后往前遍历
    for step in reversed(range(len(rewards))):
        # 如果是最后一步，使用外部传入的 next_value
        if step == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[step + 1]
            
        delta = rewards[step] + gamma * next_val * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def evaluate_policy(env, policy, eval_episodes=5):
    """
    评估循环：使用确定性策略运行若干回合，计算平均奖励。
    用于监控模型真实性能。
    """
    policy.eval()
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(Config.DEVICE)
            with torch.no_grad():
                # 开启 deterministic=True，去除随机性
                action, _, _ = policy.get_action(state_tensor, deterministic=True)
            action_np = action.cpu().numpy()[0]
            state, reward, done, _, _ = env.step(action_np)
            avg_reward += reward
    policy.train()
    return avg_reward / eval_episodes

def train(args):
    run_name = f"rl_shield_ppo_{int(time.time())}"
    log_dir = os.path.join("runs", run_name)
    writer = SummaryWriter(log_dir)
    print(f"=== 启动 TensorBoard ===\n运行命令: tensorboard --logdir={log_dir}")
    
    # 环境与模型初始化
    metrics = MetricsCalculator()
    env = RLShieldEnv(metrics)
    policy = ActorCritic(env.state_dim, env.action_space.shape[0]).to(Config.DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=Config.LR, eps=1e-5)
    
    # 学习率调整器 (Linear Decay)
    num_updates = 200 # 增加训练轮数
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=num_updates)
    
    best_reward = -float('inf')
    global_step = 0
    
    # PPO 参数
    num_steps = 128      # 每次 Rollout 收集的步数
    batch_size = num_steps
    minibatch_size = 32
    ppo_epochs = 4
    
    for update in range(1, num_updates + 1):
        # --- 1. 数据收集 (Rollout) ---
        states, actions, log_probs, rewards, masks, values = [], [], [], [], [], []
        state, _ = env.reset()
        
        for step in range(num_steps):
            global_step += 1
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(Config.DEVICE)
            
            with torch.no_grad():
                action, log_prob, value = policy.get_action(state_tensor)
                
            action_np = action.cpu().numpy()[0]
            next_state, reward, done, _, info = env.step(action_np)
            
            states.append(state_tensor)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value.item())
            masks.append(0.0 if done else 1.0)
            
            state = next_state
            if done:
                state, _ = env.reset()
                
        # 计算下一个状态的 Value 用于 GAE
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(state).unsqueeze(0).to(Config.DEVICE)
            next_value = policy.get_value(next_state_tensor).item()
            
        returns = compute_gae(rewards, values, masks, next_value, Config.GAMMA, Config.GAE_LAMBDA)
        
        # 转换为 Tensor
        returns = torch.tensor(returns).float().to(Config.DEVICE)
        states = torch.cat(states)
        actions = torch.cat(actions)
        log_probs = torch.cat(log_probs)
        values = torch.tensor(values).float().to(Config.DEVICE)
        advantages = returns - values
        
        # Flatten batches
        b_states = states.view(-1, env.state_dim)
        b_actions = actions.view(-1, env.action_space.shape[0])
        b_log_probs = log_probs.view(-1)
        b_advantages = advantages.view(-1)
        b_returns = returns.view(-1)
        b_values = values.view(-1)

        # --- 2. PPO 更新 ---
        clip_fracs = []
        for epoch in range(ppo_epochs):
            # 获取随机索引
            b_inds = np.arange(batch_size)
            np.random.shuffle(b_inds)
            
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                
                new_log_prob, new_value, entropy = policy.evaluate(b_states[mb_inds], b_actions[mb_inds])
                new_value = new_value.view(-1)
                
                logratio = new_log_prob - b_log_probs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    # 计算 KL 散度用于监控
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs += [((ratio - 1.0).abs() > Config.CLIP_EPS).float().mean().item()]
                
                # 优势归一化
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy Loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - Config.CLIP_EPS, 1 + Config.CLIP_EPS)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value Loss
                v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()
                
                # Entropy Loss
                entropy_loss = entropy.mean()
                
                loss = pg_loss + 0.5 * v_loss - 0.01 * entropy_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5) # 梯度裁剪
                optimizer.step()
        
        # 学习率衰减
        lr_scheduler.step()
        
        # --- 3. 记录与评估 ---
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("charts/mean_reward_rollout", np.mean(rewards), global_step)
        
        print(f"Update {update}/{num_updates} | Rollout Reward: {np.mean(rewards):.4f} | Loss: {loss.item():.4f}")
        
        # 定期评估 (每 10 轮)
        if update % 10 == 0:
            eval_reward = evaluate_policy(env, policy)
            writer.add_scalar("charts/eval_reward", eval_reward, global_step)
            print(f"--> Evaluation Result: {eval_reward:.4f}")
            
            # 保存最佳模型
            if eval_reward > best_reward:
                best_reward = eval_reward
                torch.save(policy.state_dict(), "rl_shield_best.pth")
                print(f"--> New Best Model Saved! (Reward: {best_reward:.4f})")
                
    # 保存最终模型
    torch.save(policy.state_dict(), "rl_shield_final.pth")
    writer.close()
    print("训练结束。请运行 `tensorboard --logdir=runs` 查看详细图表。")

def inference(args):
    print(f"=== RL-Shield 推理模式: '{args.query}' ===")
    metrics = MetricsCalculator()
    env = RLShieldEnv(metrics)
    policy = ActorCritic(env.state_dim, env.action_space.shape[0]).to(Config.DEVICE)
    
    model_path = "rl_shield_best.pth" if os.path.exists("rl_shield_best.pth") else "rl_shield_final.pth"
    if os.path.exists(model_path):
        policy.load_state_dict(torch.load(model_path))
        print(f"已加载模型: {model_path}")
    else:
        print("未找到模型文件，使用随机参数。")
        
    state, _ = env.reset(query=args.query)
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(Config.DEVICE)
    
    with torch.no_grad():
        # 推理时使用确定性动作
        action, _, _ = policy.get_action(state_tensor, deterministic=True)
        
    temp_norm, topp_norm, topk_norm = action.cpu().numpy()[0]
    
    # 参数反归一化
    temperature = Config.TEMP_MIN + (Config.TEMP_MAX - Config.TEMP_MIN) * temp_norm
    top_p = Config.TOP_P_MIN + (Config.TOP_P_MAX - Config.TOP_P_MIN) * topp_norm
    top_k = int(Config.TOP_K_MAX * topk_norm) + 1
    
    print("\n[防御策略建议]")
    print(f"Temperature: {temperature:.4f}")
    print(f"Top-P:       {top_p:.4f}")
    print(f"Top-K:       {top_k}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference"])
    parser.add_argument("--query", type=str, default="How to build a bomb")
    args = parser.parse_args()
    
    if args.mode == "train":
        train(args)
    else:
        inference(args)
