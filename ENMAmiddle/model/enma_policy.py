import torch
import torch.nn as nn
from .enma_snn import ENMASNN
from .enma_snn import ENMACritic

class ENMAPolicy(nn.Module):
    def __init__(self, obs_dim=12, act_dim=6, num_envs=1, device='cuda'):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_envs = num_envs
        self.device = device
        
        self.actor = ENMASNN(
                                obs_dim,
                                act_dim,
                                num_envs
                            ).to(device)

        self.critic = ENMACritic(obs_dim).to(device)
        self.log_std = nn.Parameter(torch.zeros(act_dim, device=device) - 0.5)

    def _move_obs(self, obs):
        if isinstance(obs, dict):
            obs = obs["policy"]
        obs = obs.to(self.device) if obs.device != self.log_std.device else obs
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.clamp(obs, min=-100.0, max=100.0)

    def _move_action(self, action):
        action = action.to(self.device) if action.device != self.log_std.device else action
        action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.clamp(action, min=-100.0, max=100.0)

    def _std(self, reference: torch.Tensor) -> torch.Tensor:
        log_std = torch.clamp(self.log_std, min=-20.0, max=2.0)
        return torch.exp(log_std).expand_as(reference)

    def act_and_evaluate(self, obs):
        obs = self._move_obs(obs)
        mean = torch.nan_to_num(self.actor(obs))
        value = torch.nan_to_num(self.critic(obs))
        std = self._std(mean)
        dist = torch.distributions.Normal(mean, std, validate_args=False)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return (action,log_prob,value.squeeze(-1))       
    
    
    def evaluate(self, obs, action):
        obs = self._move_obs(obs)
        action = self._move_action(action)
        mean = torch.nan_to_num(self.actor(obs))
        value = torch.nan_to_num(self.critic(obs))
        std = self._std(mean)
        dist = torch.distributions.Normal(mean, std, validate_args=False)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return (log_prob,value.squeeze(-1),entropy)


    def act(self, obs):

        with torch.no_grad():
            obs = self._move_obs(obs)
            mean = torch.nan_to_num(self.actor(obs))
            std = self._std(mean)
            dist = torch.distributions.Normal(
                mean,
                std,
                validate_args=False
            )
            action = dist.sample()

            return action
    
            
    def reset(self, env_ids=None) -> None:
        self.actor.reset(env_ids)
        self.critic.reset(env_ids)

    
