import torch
import torch.nn as nn
from torch.distributions import Normal
from config import Config

class ActorCritic(nn.Module):
    """
    PPO Policy Network (Actor) and Value Network (Critic).
    """
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Shared feature extraction layer (Optional)
        self.base = nn.Sequential(
            nn.Linear(state_dim, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.ReLU()
        )
        
        # Actor Head: Outputs action Mean
        # Output dimension is 3 (T, P, K)
        # Use Sigmoid to ensure output is in [0, 1], corresponding to mapping logic in env.py
        self.actor_mean = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM, action_dim),
            nn.Sigmoid() 
        )
        
        # Action Log Std, learnable parameter
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic Head: Evaluates state value
        self.critic = nn.Linear(Config.HIDDEN_DIM, 1)
        
    def forward(self, state):
        x = self.base(state)
        return x

    def get_action(self, state):
        x = self.base(state)
        mean = self.actor_mean(x)
        std = self.actor_logstd.exp().expand_as(mean)
        
        dist = Normal(mean, std)
        action = dist.sample()
        
        # Clamp to [0, 1] range as our mapping relies on 0-1
        action = torch.clamp(action, 0.0, 1.0)
        
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, action_log_prob, self.critic(x)

    def evaluate(self, state, action):
        x = self.base(state)
        mean = self.actor_mean(x)
        std = self.actor_logstd.exp().expand_as(mean)
        
        dist = Normal(mean, std)
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        
        return action_log_prob, self.critic(x), dist_entropy
