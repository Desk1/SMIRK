import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
import random


class Sample:
    def __init__(self, value: torch.Tensor, fitness_score: int = -1):
        self.value = value
        self.fitness_score = fitness_score

class VectorizedPopulation:
    def __init__(
            self,
            all_ws: torch.Tensor,
            all_logits: torch.Tensor,
            population_size: int,
            target_label: int,
            p_std_ce: float = 1.0 #bound for p_space_bound mean+-x*std; set 0. to unbound
        ):

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

        all_ws = all_ws[:population_size]
        self.all_w_mins, self.all_w_maxs = compute_p_bounds(all_ws, p_std_ce)

        all_logits = all_logits[:population_size]
        all_prediction = F.log_softmax(all_logits, dim=1)[:, target_label]
        topk_conf, topk_ind = torch.topk(all_prediction, population_size, dim=0, largest=True, sorted=True)

        self.population = all_ws[topk_ind].detach().clone()
        self.fitness_scores = topk_conf

    def clip(self, w):
        assert w.ndim == 2
        return torch.max(torch.min(w, self.all_w_maxs), self.all_w_mins)
    
    def clip_array(self, inputs):
        clipped = np.clip(inputs, self.all_w_mins.cpu().detach().numpy()[0], self.all_w_maxs.cpu().detach().numpy()[0])
        return clipped

    def find_elite(self, index=0):
        self.fitness_scores, indices = torch.sort(self.fitness_scores, dim=0, descending=True)
        self.population = self.population[indices]
        return Sample(self.population[index].detach().clone(), self.fitness_scores[index].item())

    def visualize_imgs(self, filename, generate_images_func, k=8):
        ws = self.population[:k]
        out = generate_images_func(ws, raw_img=True)
        vutils.save_image(out, filename)

class VectorizedPopulationMirror(VectorizedPopulation):
    def __init__(
        self,
        *args,
        mutation_prob: float = 0.1,
        mutation_ce: float = 0.1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.mutation_prob = mutation_prob
        self.mutation_ce = mutation_ce

    def compute_fitness(self):
        bs = 50
        scores = []
        for i in range(0, len(self.population), bs):
            data = self.population[i:i+bs]
            scores.append(self.compute_fitness_func(data))
        self.fitness_scores = torch.cat(scores, dim=0)
        assert self.fitness_scores.ndim == 1 and self.fitness_scores.shape[0] == len(self.population)

    def __get_parents(self, k):
        weights = F.softmax(self.fitness_scores, dim=0).tolist()
        parents_ind = random.choices(list(range(len(weights))), weights=weights, k=2*k)
        parents1_ind = parents_ind[:k]
        parents2_ind = parents_ind[k:]

        return parents1_ind, parents2_ind

    def __crossover(self, parents1_ind, parents2_ind):
        parents1_fitness_scores = self.fitness_scores[parents1_ind]
        parents2_fitness_scores = self.fitness_scores[parents2_ind]
        p = (parents1_fitness_scores / (parents1_fitness_scores + parents2_fitness_scores)).unsqueeze(1)  # size: N, 1
        parents1 = self.population[parents1_ind].detach().clone()  # size: N, 512
        parents2 = self.population[parents2_ind].detach().clone()  # size: N, 512
        mask = torch.rand_like(parents1)
        mask = (mask < p).float()
        return mask*parents1 + (1.-mask)*parents2

    def __mutate(self, children):
        mask = torch.rand_like(children)
        mask = (mask < self.mutation_prob).float()
        children = self.apply_noise_func(children, mask, self.mutation_ce)
        return self.clip_func(children)

    def produce_next_generation(self, elite):
        parents1_ind, parents2_ind = self.__get_parents(len(self.population)-1)
        children = self.__crossover(parents1_ind, parents2_ind)
        mutated_children = self.__mutate(children)
        self.population = torch.cat((elite.value.unsqueeze(0), mutated_children), dim=0)
        self.compute_fitness()
