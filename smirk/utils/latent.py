# smirk/utils/latent.py - latent space utils

# API
# -------------------------------------------
# clip_quantile_bound(inputs, mins, maxs)    -> Tensor   — clamp elements to bounds

import torch

#########################
# Latent space handling #
#########################

def clip_quantile_bound(inputs, all_mins, all_maxs):
    clipped = torch.max(torch.min(inputs, all_maxs), all_mins)
    return clipped