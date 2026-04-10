- redo documentation for utils

- check sample weighting behaviour
    kl_loss = (kl_loss * topk_weight).sum()

    supposedly not actually achieving sample level reweighting but a global scalar reweight