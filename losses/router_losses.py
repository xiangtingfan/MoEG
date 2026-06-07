import torch


def router_balance_loss(router_weights_all):
    if len(router_weights_all) == 0:
        return torch.tensor(0.0)

    loss = 0.0
    for weights in router_weights_all:
        expert_usage = weights.mean(dim=0)
        num_experts = weights.shape[-1]
        target = torch.ones_like(expert_usage) / num_experts
        loss = loss + torch.sum((expert_usage - target) ** 2)

    return loss / len(router_weights_all)


def router_entropy_loss(router_weights_all):
    if len(router_weights_all) == 0:
        return torch.tensor(0.0)

    loss = 0.0
    for weights in router_weights_all:
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1)
        loss = loss - entropy.mean()

    return loss / len(router_weights_all)
