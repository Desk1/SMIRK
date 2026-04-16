# smirk/utils/latent.py - latent space utils

"""
API
-------------------------------------------
clip_quantile_bound(inputs, mins, maxs)    # clamp elements to bounds
"""

import torch
import torch.nn as nn


def compute_p_bounds(all_ws, p_std_ce):
    # compute bound in p space
    invert_lrelu = nn.LeakyReLU(negative_slope=5.)
    lrelu = nn.LeakyReLU(negative_slope=0.2)

    all_ps = invert_lrelu(all_ws)
    all_p_means = torch.mean(all_ps, dim=0, keepdim=True)
    all_p_stds = torch.std(all_ps, dim=0, keepdim=True, unbiased=False)
    all_p_mins = all_p_means - p_std_ce * all_p_stds
    all_p_maxs = all_p_means + p_std_ce * all_p_stds

    all_w_mins = lrelu(all_p_mins)
    all_w_maxs = lrelu(all_p_maxs)

    return all_w_mins, all_w_maxs

def clip_quantile_bound(inputs, all_mins, all_maxs):
    clipped = torch.max(torch.min(inputs, all_maxs), all_mins)
    return clipped
