import torch 
from types import SimpleNamespace

class RolloutBuffer:
    def __init__(self):
       self.obs = []
       self.actions = []
       self.log_probs = []
       self.values = []
       self.rewards = []
       self.dones = []

    def add(self, obs, action, log_prob, value, reward, done):

        self.obs.append(obs.detach().cpu())
        self.actions.append(action.detach().cpu())
        self.log_probs.append(log_prob.detach().cpu())              
        self.values.append(value.detach().cpu())
        self.rewards.append(reward.detach().cpu())
        self.dones.append(done.detach().cpu())

    
    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
    
    def get(self):
        return (
            torch.stack(self.obs),
            torch.stack(self.actions),
            torch.stack(self.log_probs),
            torch.stack(self.values),
            torch.stack(self.rewards),
            torch.stack(self.dones)
        )
    

    def minibatches(
    self,
    returns,
    advantages,
    batch_size=256,
    device=None):

        obs = torch.cat(self.obs, dim=0)
        actions = torch.cat(self.actions, dim=0)
        log_probs = torch.cat(self.log_probs, dim=0)

        values = torch.cat(self.values, dim=0)

        returns = returns.reshape(-1)
        advantages = advantages.reshape(-1)

        if device is None:
            device = obs.device

        obs = obs.to(device)
        actions = actions.to(device)
        log_probs = log_probs.to(device)
        returns = returns.to(device)
        advantages = advantages.to(device)

        num_samples = obs.shape[0]

        indices = torch.randperm(
            num_samples,
            device=obs.device
        )

        for start in range(
            0,
            num_samples,
            batch_size
        ):

            end = start + batch_size

            batch_idx = indices[start:end]

            yield SimpleNamespace(
                obs=obs[batch_idx],
                action=actions[batch_idx],
                log_prob=log_probs[batch_idx],
                return_=returns[batch_idx],
                advantage=advantages[batch_idx],
            )
