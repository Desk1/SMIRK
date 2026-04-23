- redo documentation for utils

- check sample weighting behaviour
    kl_loss = (kl_loss * topk_weight).sum()

    supposedly not actually achieving sample level reweighting but a global scalar reweight

- document long-tailed_surrogate_training line 701, 0.15 > int bug in report

- move attack output / all output to hydra timestamped 'outputs' experiment folder
