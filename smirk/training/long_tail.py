import torch


def build_weight_k(all_logits, beta):
    
    topk_indices = torch.topk(all_logits, k=10, dim=1)[1]
    
    flattened_indices = topk_indices.view(-1)
    label_counts = torch.bincount(flattened_indices, minlength=all_logits.shape[1])
    
    weights = torch.zeros_like(label_counts, dtype=torch.float)
    max_count = torch.max(label_counts)
    min_count = torch.min(label_counts)

    for idx, count in enumerate(label_counts):
        weights[idx] = (1-beta)/(1-beta**(count+1)) # (1-beta)/(1-beta**count)
    
    return weights