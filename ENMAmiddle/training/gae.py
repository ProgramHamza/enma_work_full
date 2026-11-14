import torch


def compute_gae(
    buffer,
    last_value,
    gamma=0.99,
    lam=0.95
):
    """
    PPO Generalized Advantage Estimation.

    Returns:
        returns
        advantages
    """

    device = last_value.device
    rewards = [reward.to(device) for reward in buffer.rewards]
    values = [value.to(device) for value in buffer.values]
    dones = [done.to(device) for done in buffer.dones]

    advantages = []
    gae = torch.zeros_like(last_value)

    next_value = last_value

    for t in reversed(range(len(rewards))):

        mask = 1.0 - dones[t].float()

        delta = (
            rewards[t]
            + gamma * next_value * mask
            - values[t]
        )

        gae = (
            delta
            + gamma * lam * mask * gae
        )

        advantages.insert(0, gae)

        next_value = values[t]

    advantages = torch.stack(advantages)

    values = torch.stack(values)

    returns = advantages + values

    return returns, advantages
