import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from env import RLShieldEnv
from agent import ActorCritic
from utils import MetricsCalculator
from config import Config

def compute_gae(rewards, values, masks, gamma=0.99, tau=0.95):
    """
    计算广义优势估计 (Generalized Advantage Estimation, GAE)。
    """
    gae = 0
    returns = []
    # 从后往前逆序计算
    for step in reversed(range(len(rewards))):
        # delta = r_t + gamma * V(s_{t+1}) * mask - V(s_t)
        # 注意：这里简化处理，假设最后一步的 next_value 为 0 (因为是单步任务或回合结束)
        if step == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[step + 1]
            
        delta = rewards[step] + gamma * next_value * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        
        # Return = Advantage + Value
        returns.insert(0, gae + values[step])
        
    return returns

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
    """
    生成器：将收集到的数据划分为 mini-batch 供 PPO 更新使用。
    """
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield (states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids], 
               returns[rand_ids], advantages[rand_ids])

def train():
    print("=== 开始 RL-Shield PPO 完整训练流程 ===")
    
    # 1. 初始化环境与模型
    metrics = MetricsCalculator()
    env = RLShieldEnv(metrics)
    
    state_dim = env.state_dim
    action_dim = env.action_space.shape[0]
    
    policy = ActorCritic(state_dim, action_dim).to(Config.DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=Config.LR)
    
    # 训练参数
    num_updates = 50       # 总共进行多少轮更新 (模拟 Config.EPOCHS 的外层含义)
    steps_per_batch = Config.BATCH_SIZE  # 每轮收集多少样本
    ppo_epochs = 4         # 每批数据在 PPO 内部更新多少次
    mini_batch_size = 16   # PPO 内部更新的 batch size
    
    # 主训练循环
    for i_update in range(num_updates):
        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        masks = []
        
        # === A. 数据收集阶段 (Rollout) ===
        # 在这个阶段，我们只与环境交互，不更新参数
        for _ in range(steps_per_batch):
            # 重置环境获取当前状态
            state, _ = env.reset()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(Config.DEVICE)
            
            # 获取动作分布和价值
            with torch.no_grad():
                action, action_log_prob, value = policy.get_action(state_tensor)
            
            # 与环境交互
            action_np = action.cpu().numpy()[0]
            next_state, reward, done, _, info = env.step(action_np)
            
            # 存储轨迹数据
            log_probs.append(action_log_prob)
            values.append(value.item()) # 存储标量
            rewards.append(reward)
            masks.append(0 if done else 1) # 如果结束了，mask为0
            
            states.append(state_tensor)
            actions.append(action)
            
        # === B. 优势计算阶段 ===
        # 计算 GAE 和 Returns
        returns = compute_gae(rewards, values, masks, Config.GAMMA, Config.GAE_LAMBDA)
        
        # 转换为 Tensor 准备更新
        returns = torch.tensor(returns).to(Config.DEVICE).unsqueeze(1)
        values_tensor = torch.tensor(values).to(Config.DEVICE).unsqueeze(1)
        log_probs = torch.cat(log_probs).detach()
        states = torch.cat(states)
        actions = torch.cat(actions)
        
        # 优势函数 A = R - V
        advantages = returns - values_tensor
        # 优势归一化 (有助于训练稳定)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # === C. PPO 更新阶段 ===
        total_actor_loss = 0
        total_critic_loss = 0
        
        for _ in range(ppo_epochs):
            for state_batch, action_batch, old_log_probs, return_batch, advantage_batch in ppo_iter(
                mini_batch_size, states, actions, log_probs, returns, advantages
            ):
                # 在新策略下评估旧动作
                # get_action 在这里可能需要稍微修改以接受 batch 或者我们手动调用 evaluate
                # 假设 agent.py 中有 evaluate 方法
                new_log_probs, state_values, dist_entropy = policy.evaluate(state_batch, action_batch)
                
                # --- PPO 核心 Loss 计算 ---
                # 1. 计算比率 r_t(\theta) = exp(new_log_prob - old_log_prob)
                ratio = torch.exp(new_log_probs - old_log_probs.detach())
                
                # 2. 计算 Surrogate Loss
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1.0 - Config.CLIP_EPS, 1.0 + Config.CLIP_EPS) * advantage_batch
                
                # PPO Clip Loss (取负号因为是梯度下降)
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 3. Value Loss (MSE)
                critic_loss = F.mse_loss(state_values, return_batch)
                
                # 4. Entropy Bonus (鼓励探索)
                entropy_loss = -0.01 * dist_entropy.mean()
                
                # 总 Loss
                loss = actor_loss + 0.5 * critic_loss + entropy_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()

        # === D. 日志记录 ===
        avg_reward = np.mean(rewards)
        print(f"Update {i_update+1}/{num_updates} | "
              f"平均奖励: {avg_reward:.4f} | "
              f"Actor Loss: {total_actor_loss:.4f} | "
              f"Critic Loss: {total_critic_loss:.4f}")
        
    print("=== 训练完成 ===")
    torch.save(policy.state_dict(), "rl_shield_policy.pth")
    print("模型已保存至 rl_shield_policy.pth")

def inference(query):
    print(f"=== RL-Shield 防御推理模式: '{query}' ===")
    
    metrics = MetricsCalculator()
    env = RLShieldEnv(metrics) # 仅用于获取 state 维度逻辑
    
    # 加载模型
    policy = ActorCritic(env.state_dim, env.action_space.shape[0]).to(Config.DEVICE)
    try:
        policy.load_state_dict(torch.load("rl_shield_policy.pth"))
        print(">> 成功加载训练好的策略模型。")
    except FileNotFoundError:
        print(">> 警告：未找到模型文件，将使用随机初始化参数进行演示。")
    
    policy.eval()
    
    # 1. 构建状态
    state, _ = env.reset(query=query)
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(Config.DEVICE)
    
    # 2. 获取动作 (推理时直接取均值，或者采样)
    with torch.no_grad():
        action, _, _ = policy.get_action(state_tensor)
        action_np = action.cpu().numpy()[0]
    
    # 3. 映射到实际参数
    temp_norm, topp_norm, topk_norm = action_np
    temperature = Config.TEMP_MIN + (Config.TEMP_MAX - Config.TEMP_MIN) * temp_norm
    top_p = Config.TOP_P_MIN + (Config.TOP_P_MAX - Config.TOP_P_MIN) * topp_norm
    top_k = int(Config.TOP_K_MAX * topk_norm) + 1
    
    print("\n[防御策略建议配置]")
    print(f"Temperature (温度): {temperature:.4f}")
    print(f"Top-P (核采样):     {top_p:.4f}")
    print(f"Top-K (候选数):     {top_k}")
    print("-" * 30)
    print("请将上述参数应用于您的 LLM generate() 函数以获得最佳防御效果。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL-Shield 训练与推理脚本")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference"], help="运行模式: train (训练) 或 inference (推理)")
    parser.add_argument("--query", type=str, default="How to make a virus?", help="推理模式下的输入查询")
    args = parser.parse_args()
    
    if args.mode == "train":
        train()
    else:
        inference(args.query)
