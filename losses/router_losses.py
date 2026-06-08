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


def _cv_squared(values, eps=1e-10):
    if values.numel() <= 1:
        return values.new_tensor(0.0)
    return values.float().var(unbiased=False) / (values.float().mean() ** 2 + eps)


def router_importance_loss(router_weights_all):
    """Classic MoE importance loss: balance the total soft gate mass per expert."""
    if len(router_weights_all) == 0:
        return torch.tensor(0.0)

    loss = 0.0
    for weights in router_weights_all:
        importance = weights.sum(dim=0)
        loss = loss + _cv_squared(importance)

    return loss / len(router_weights_all)


def router_load_loss(router_weights_all):
    """Classic MoE load loss: balance the number of samples routed to each expert."""
    if len(router_weights_all) == 0:
        return torch.tensor(0.0)

    loss = 0.0
    for weights in router_weights_all:
        load = (weights > 0).float().sum(dim=0)
        loss = loss + _cv_squared(load)

    return loss / len(router_weights_all)


def compute_mmd_loss(features, subject_ids):
    """Simplified source-domain MMD: minimize pairwise subject feature mean distance."""
    unique_subjects = torch.unique(subject_ids)
    if len(unique_subjects) < 2:
        return torch.tensor(0.0, device=features.device)

    mmd_loss = features.new_tensor(0.0)
    count = 0

    for i, subject_i in enumerate(unique_subjects):
        for j, subject_j in enumerate(unique_subjects):
            if i >= j:
                continue

            features_i = features[subject_ids == subject_i]
            features_j = features[subject_ids == subject_j]

            if len(features_i) > 0 and len(features_j) > 0:
                mean_i = features_i.mean(dim=0)
                mean_j = features_j.mean(dim=0)
                mmd_loss = mmd_loss + torch.norm(mean_i - mean_j, p=2)
                count += 1

    return mmd_loss / max(count, 1)


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, epsilon=1e-5):
    n_samples = int(source.size(0)) + int(target.size(0))
    if n_samples <= 1:
        return source.new_zeros((n_samples, n_samples))

    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    l2_distance = ((total0 - total1) ** 2).sum(dim=2) + epsilon

    if fix_sigma is not None:
        bandwidth = source.new_tensor(float(fix_sigma))
    else:
        bandwidth = torch.sum(l2_distance.detach()) / max(n_samples ** 2 - n_samples, 1)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-l2_distance / bandwidth_temp.clamp_min(epsilon)) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def cmmd(source, target, source_labels, target_labels, num_classes, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    if source.size(0) == 0 or target.size(0) == 0:
        return source.new_tensor(0.0)

    source_labels = source_labels.view(-1, 1).long()
    target_labels = target_labels.view(-1, 1).long()
    source_one_hot = source.new_zeros((source_labels.size(0), num_classes)).scatter_(1, source_labels, 1.0)
    target_one_hot = target.new_zeros((target_labels.size(0), num_classes)).scatter_(1, target_labels, 1.0)

    batch_size_source = int(source_one_hot.size(0))
    batch_size_target = int(target_one_hot.size(0))
    kernels = gaussian_kernel(
        source,
        target,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma,
    )

    xx = kernels[:batch_size_source, :batch_size_source]
    yy = kernels[batch_size_source:, batch_size_source:]
    xy = kernels[:batch_size_source, batch_size_source:]
    yx = kernels[batch_size_source:, :batch_size_source]

    loss_xx = torch.mean(torch.mm(source_one_hot, source_one_hot.t()) * xx)
    loss_yy = torch.mean(torch.mm(target_one_hot, target_one_hot.t()) * yy)
    loss_xy = torch.mean(torch.mm(source_one_hot, target_one_hot.t()) * xy)
    loss_yx = torch.mean(torch.mm(target_one_hot, source_one_hot.t()) * yx)
    return loss_xx + loss_yy - loss_xy - loss_yx


def compute_subject_cmmd_loss(features, labels, subject_ids, num_classes):
    """Pairwise CMMD across source subjects inside one training batch."""
    unique_subjects = torch.unique(subject_ids)
    if len(unique_subjects) < 2:
        return features.new_tensor(0.0)

    loss = features.new_tensor(0.0)
    count = 0
    for i, subject_i in enumerate(unique_subjects):
        for j, subject_j in enumerate(unique_subjects):
            if i >= j:
                continue

            mask_i = subject_ids == subject_i
            mask_j = subject_ids == subject_j
            if mask_i.any() and mask_j.any():
                loss = loss + cmmd(
                    features[mask_i],
                    features[mask_j],
                    labels[mask_i],
                    labels[mask_j],
                    num_classes=num_classes,
                )
                count += 1

    return loss / max(count, 1)
