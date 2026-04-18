import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from tqdm import tqdm

from smirk.attacks.base import BaseAttack, AttackResult
from smirk.attacks.population import Sample
from smirk.utils.image import crop_and_resize, normalize


class SMILEWhiteboxAttack(BaseAttack):
    def __init__(
            self,
            *args,
            init_idex: int = 0,
            **kwargs
        ):
        super().__init__(*args, **kwargs)

        elite = self.population.find_elite(init_idex)
        self.L = elite.value.unsqueeze(0).clone().to(self.device)
        self.L.requires_grad = True

        self.optimizer = optim.Adam([self.L, ], lr=self.learning_rate, betas=[0.9, 0.999], eps=1e-8)

        self.criterion = nn.CrossEntropyLoss()

    def adjust_lr(self, initial_lr, epoch, rampdown=0.25, rampup=0.05):
        # from https://github.com/rosinality/style-based-gan-pytorch/blob/master/projector.py#L45
        t = epoch / self.epochs
        lr_ramp = min(1, (1 - t) / rampdown)
        lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
        lr_ramp = lr_ramp * min(1, t / rampup)

        lr = initial_lr * lr_ramp

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr
    
    def run(self, target_label: int) -> AttackResult:
        TARGET = torch.LongTensor([target_label]).to(self.device)

        for epoch in tqdm(range(self.epochs + 1)):
            _lr = self.adjust_lr(self.learning_rate, epoch)
            img = crop_and_resize(
                self.generator(self.L),
                self.target_model_spec.name,
                self.target_model_spec.resolution
            )

            self.optimizer.zero_grad()
            self.generator.zero_grad()

            assert img.ndim == 4

            self.target_model.eval()
            if self.target_model_spec.name == 'inception_v3':
                outputs, _, _, _ = self.target_model(normalize(img*255., self.target_model_spec.name), 0)
            else:
                outputs, _ = self.target_model(normalize(img*255., self.target_model_spec.name))
            
            loss = self.criterion(outputs, TARGET)

            loss.backward()
            self.optimizer.step()

            self.L.data = self.population.clip(self.L.data)

            with torch.no_grad():
                img1 = self.generate_images(self.L)

                outputs = self.test_model(normalize(crop_and_resize(img1, self.test_model_spec.name, self.test_model_spec.resolution)*255., self.test_model_spec.name))

                if self.test_model_spec.name == 'sphere20a':
                    outputs = outputs[0]

                logits_softmax = F.log_softmax(outputs, dim=1)[:, target_label]

                rank_of_label = torch.sum(F.log_softmax(outputs, dim=1) > logits_softmax)

                if epoch % 10 == 0 or epoch == self.epochs-1:
                    self.writer.add_scalar('Target Score', loss.item(), epoch+1)
                    self.writer.add_scalar('Evaluation Score', logits_softmax.item(), epoch+1)
                    self.writer.add_scalar('Rank of label', rank_of_label.item(), epoch+1)

                    self.writer.add_image('Generated Image', img1.squeeze(), global_step=epoch)

        return AttackResult(
            latent_vector = self.L.detach().clone(),
            fitness_score = logits_softmax.item(),
            generated_image = img1.squeeze()
        )


class MirrorWhiteboxAttack(BaseAttack):
    def __init__(
            self,
            *args,
            init_idex: int = 0,
            **kwargs
        ):
        super().__init__(*args, **kwargs)

        elite = self.population.find_elite(init_idex)
        self.L = elite.value.unsqueeze(0).clone().to(self.device)
        self.L.requires_grad = True

        self.optimizer = optim.Adam([self.L, ], lr=self.learning_rate, betas=[0.9, 0.999], eps=1e-8)

        self.criterion = nn.CrossEntropyLoss()
    
    def run(self, target_label: int) -> AttackResult:
        TARGET = torch.LongTensor([target_label]).to(self.device)

        for epoch in tqdm(range(self.epochs + 1)):
            img = crop_and_resize(
                self.generator(self.L),
                self.target_model_spec.name,
                self.target_model_spec.resolution
            )

            self.optimizer.zero_grad()
            self.generator.zero_grad()

            assert img.ndim == 4

            self.target_model.eval()
            outputs = self.target_model(normalize(img*255., self.target_model_spec.name))

            if self.target_model_spec.name == 'sphere20a':
                outputs = outputs[0]
            
            loss = self.criterion(outputs, TARGET)

            loss.backward()
            self.optimizer.step()

            self.L.data = self.population.clip(self.L.data)

            with torch.no_grad():
                img1 = self.generate_images(self.L)

                outputs = self.test_model(normalize(crop_and_resize(img1, self.test_model_spec.name, self.test_model_spec.resolution)*255., self.test_model_spec.name))

                if self.test_model_spec.name == 'sphere20a':
                    outputs = outputs[0]

                logits_softmax = F.log_softmax(outputs, dim=1)[:, target_label]

                rank_of_label = torch.sum(F.log_softmax(outputs, dim=1) > logits_softmax)

                if epoch % 10 == 0 or epoch == self.epochs-1:
                    self.writer.add_scalar('Target Score', loss.item(), epoch+1)
                    self.writer.add_scalar('Evaluation Score', logits_softmax.item(), self.epoch+1)
                    self.writer.add_scalar('Rank of label', rank_of_label.item(), epoch+1)

                    self.writer.add_image('Generated Image', img1.squeeze(), global_step=epoch)

        return AttackResult(
            latent_vector = self.L.detach().clone(),
            fitness_score = logits_softmax.item(),
            generated_image = img1.squeeze()
        )