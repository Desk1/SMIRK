import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def calculate_loss(
    model: nn.Module,
    data: Tensor,
    target: Tensor,
    topk_weight: Tensor,
    num_experts: int,
    lambda_diversity: float,
    lambda_ce: float,
    temperature: float = 0.5,
):  
    output = None
    expert_outputs = None
    aux_ouput = None
    aux_expert_outputs = None
    if hasattr(model, "AuxLogits"): # inception_v3
        output, expert_outputs, aux_ouput, aux_expert_outputs = model(data, 1)
    else:
        output, expert_outputs = model(data)

    # calculate losses
    kl_loss = 0
    ce_loss = 0
    d_loss = 0
    for i in range(num_experts):
        kl_loss += kl_distillation_loss(target, expert_outputs[i], temperature)
        ce_loss += ce_cross_entropy_loss(target, expert_outputs[i])

        stabilised_expert_output = torch.clamp(expert_outputs[i] / temperature, -20, 20)
        d_loss += 1.0 / diversity_loss(output, stabilised_expert_output, temperature)

        if aux_expert_outputs:
            kl_loss += 0.3 * kl_distillation_loss(target, aux_expert_outputs[i], temperature)
            ce_loss += ce_cross_entropy_loss(target, aux_expert_outputs[i])

            aux_stabilised_expert_output = torch.clamp(aux_expert_outputs[i] / temperature, -20, 20)
            d_loss += 0.3 * (1.0 / diversity_loss(aux_ouput, aux_stabilised_expert_output, temperature))

    # top-k sample reweighting
    # todo - incorrect calcualtion? doesnt this just multiply kl_loss (scalar) by sum(topk_weight), resulting in a global scalar reweighting instead of sample level reweighting
    kl_loss = (kl_loss * topk_weight).sum()
    ce_loss = (ce_loss * topk_weight).sum()

    # hyperparameter weighting
    ce_loss = lambda_ce * ce_loss
    d_loss = lambda_diversity * d_loss

    total_loss = kl_loss + ce_loss + d_loss
    return total_loss, kl_loss, ce_loss, d_loss


def kl_distillation_loss(
    target: Tensor,
    expert_output: Tensor,
    temperature: float
):
    """KL divergence for knowledge distillation"""

    return F.kl_div(
        F.log_softmax(expert_output / temperature, dim=1),
        F.softmax(target / temperature, dim=1),
        reduction='batchmean'
    )


def ce_cross_entropy_loss(
    target: Tensor,
    expert_output: Tensor
):
    _, predicted = torch.max(target, 1)
    criterion = nn.CrossEntropyLoss()

    return criterion[expert_output, predicted]

def diversity_loss(
    raw_output: Tensor,
    stabilised_expert_output: Tensor,
    temperature: float
):
    return F.kl_div(
        F.log_softmax(raw_output / temperature, dim=1),
        F.softmax(stabilised_expert_output / temperature, dim=1),
        reduction='batchmean'
    )
