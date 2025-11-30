import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from config import Config

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    正交初始化 (Orthogonal Initialization)。
    这是 PPO 稳定收敛的关键技巧，比默认的 Xavier/Kaiming 初始化在强化学习中效果更好。
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    """
    增强版 PPO 策略网络 (Actor) 和价值网络 (Critic)。
    """
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Critic 网络: 评估状态价值 V(s)
        # 输出层 std 设置为 1.0
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, Config.HIDDEN_DIM)),
            nn.Tanh(),
            layer_init(nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM)),
            nn.Tanh(),
            layer_init(nn.Linear(Config.HIDDEN_DIM, 1), std=1.0),
        )
        
        # Actor 网络: 输出动作均值 (Mean)
        # 输出层 std 设置为 0.01，这有助于在训练初期让策略接近随机输出，防止过早收敛到局部最优
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_dim, Config.HIDDEN_DIM)),
            nn.Tanh(),
            layer_init(nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM)),
            nn.Tanh(),
            layer_init(nn.Linear(Config.HIDDEN_DIM, action_dim), std=0.01),
            nn.Sigmoid() # 确保输出在 [0, 1] 之间，对应环境中的归一化参数范围
        )
        
        # 动作对数标准差 (Log Std)，可学习参数
        # 初始化为 -0.5 (即 std ≈ 0.6)，给与一定的初始探索噪声
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim) - 0.5)
        
    def get_value(self, x):
        """仅获取状态价值 V(s)"""
        return self.critic(x)

    def get_action(self, x, deterministic=False):
        """
        获取动作。
        参数:
            x: 状态张量
            deterministic: 
                - True: 直接返回均值 (用于推理/评估，消除随机性，保证防御策略稳定)
                - False: 从分布中采样 (用于训练，保持探索性)
        返回:
            action, log_prob, entropy (如果 deterministic=True，后两者为 None)
        """
        action_mean = self.actor_mean(x)
        
        if deterministic:
            return action_mean, None, None
            
        # 扩展 std 以匹配 batch size
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        # 构建正态分布
        probs = Normal(action_mean, action_std)
        
        # 采样动作
        action = probs.sample()
        
        # 截断动作以符合环境物理意义 [0, 1]
        # 注意：虽然 Sigmoid 限制了均值，但高斯采样可能会溢出，所以必须 clamp
        action = torch.clamp(action, 0.0, 1.0)
        
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)

    def evaluate(self, x, action):
        """
        用于 PPO 更新阶段：计算给定状态 batch 和动作 batch 下的统计量。
        这是计算 Loss 的核心部分。
        """
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        # 计算 log_prob (sum over action dimensions)
        action_log_prob = probs.log_prob(action).sum(1)
        
        # 计算 entropy (用于鼓励探索)
        dist_entropy = probs.entropy().sum(1)
        
        # 计算 value
        state_value = self.critic(x)
        
        return action_log_prob, state_value, dist_entropy
