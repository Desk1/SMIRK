import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from tqdm import tqdm
import nevergrad as ng
from typing import List, Dict

from smirk.attacks.base import BaseAttack, AttackResult
from smirk.attacks.population import Sample
from smirk.utils.image import crop_and_resize, normalize


class SMILEBlackboxAttack(BaseAttack):
    def __init__(
            self,
            *args,
            elite_vector: torch.Tensor,
            budget: int,
            optimizer_strategy: str,
            **kwargs
        ):
        super().__init__(*args, **kwargs)

        if elite_vector is None:
            raise RuntimeError(f"Failed to load elite starting point")

        elite_vector = elite_vector.squeeze()
        self.L = elite_vector.cpu().numpy()
        self.elite = elite_vector

        self.budget = budget

        parametrization = ng.p.Array(init=self.L).set_mutation(sigma=1)
        self.optimizer = ng.optimizers.registry[optimizer_strategy](parametrization=parametrization, budget=budget)

    def compute_fitness(self, w, target_label):
            img = self.generate_images(w)
            img = crop_and_resize(img, self.target_model_spec.name, self.target_model_spec.resolution)

            assert img.ndim == 4

            if self.target_model_spec.name == 'sphere20a':
                pred = F.log_softmax(self.target_model(normalize(img*255., self.target_model_spec.name))[0], dim=1)
            else:
                pred = F.log_softmax(self.target_model(normalize(img*255., self.target_model_spec.name)), dim=1)

            score = pred[:, target_label]
            return score 

    def run(self, target_label: int):
        self.target_model.eval()
        self.test_model.eval()

        score_0 = self.compute_fitness(self.elite.unsqueeze(0), target_label) # 
        img_0 = self.generate_images(self.elite.unsqueeze(0)) #
        outputs_0 = self.test_model(normalize(crop_and_resize(img_0, self.test_model_spec.name, self.test_model_spec.resolution)*255., self.test_model_spec.name))
        if self.test_model_spec.name == 'sphere20a':
            outputs_0 = outputs_0[0]
        logits_softmax_0 = F.log_softmax(outputs_0, dim=1)[:, target_label]
        rank_of_label_0 = torch.sum(F.log_softmax(outputs_0, dim=1) > logits_softmax_0)

        self.writer.add_scalar('Target Score', score_0.item(), 0)
        self.writer.add_scalar('Evaluation Score', logits_softmax_0.item(), 0)
        self.writer.add_scalar('Rank of label', rank_of_label_0.item(), 0)
        
        self.results["original"] = AttackResult(
            latent_vector = self.elite.cpu().clone(),
            fitness_score = logits_softmax_0.item(),
            generated_image = img_0
        )

        T = 1
        for r in tqdm(range(self.budget)):
            ng_data = [self.optimizer.ask() for _ in range(T)]

            for index in range(T):
                clipped_ng_data = self.population.clip_array(ng_data[index].value)
                ng_data[index].value[:] = clipped_ng_data
            
            score = [self.compute_fitness(torch.Tensor(ng_data[i].value).type(torch.float32).unsqueeze(0), target_label) for i in range(T)] #

            for z, l in zip(ng_data, score):
                self.optimizer.tell(z, l.item()*(-1.))
        
            recommendation = self.optimizer.provide_recommendation()
            recommendation = torch.Tensor(recommendation.value).type(torch.float32) #

            recommendation = recommendation.unsqueeze(0)
            img = self.generate_images(recommendation)

            outputs = self.test_model(normalize(crop_and_resize(img, self.test_model_spec.name, self.test_model_spec.resolution)*255., self.test_model_spec.name))
            if self.test_model_spec.name == 'sphere20a':
                outputs = outputs[0]
            logits_softmax = F.log_softmax(outputs, dim=1)[:, target_label]

            # intermediate checkpoints
            if r % 500 == 0:
                save_w = Sample(recommendation.detach().clone(), logits_softmax.item())
                self.results[str(r)] = AttackResult(
                    latent_vector = recommendation.detach().clone(),
                    fitness_score = logits_softmax.item(),
                    generated_image = img
                )

            rank_of_label = torch.sum(F.log_softmax(outputs, dim=1) > logits_softmax)

            if r % 10 == 0 or r == self.budget-1:
                self.writer.add_scalar('Target Score', score[0].item(), r+1)
                self.writer.add_scalar('Evaluation Score', logits_softmax.item(), r+1)
                self.writer.add_scalar('Rank of label', rank_of_label.item(), r+1)
                self.writer.add_image('Generated Image', img.squeeze(), global_step=r)
            
        self.results["final"] = AttackResult(
            latent_vector = recommendation.detach().clone(),
            fitness_score = logits_softmax.item(),
            generated_image = img
        )


