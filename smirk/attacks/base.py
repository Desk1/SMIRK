import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
from abc import ABC, abstractmethod

from smirk.attacks.population import VectorizedPopulation
from smirk.models.registry import ModelSpec
from smirk.genforce import my_get_GD


@dataclass
class AttackResult:
    latent_vector: torch.Tensor
    fitness_score: float
    generated_image: torch.Tensor

@dataclass  
class AttackMetrics:
    top1_acc: float
    top5_acc: float
    target_confidence: float
    

class BaseAttack(ABC):
    def __init__(
            self,
            target_model: nn.Module,
            target_model_spec: ModelSpec,
            test_model: nn.Module,
            test_model_spec: ModelSpec,
            generator: my_get_GD.Fake_G,
            writer: SummaryWriter,
            device: torch.device,
            population: VectorizedPopulation,
            epochs: int,
            learning_rate: float
    ):
        self.target_model = target_model
        self.generator = generator
        self.writer = writer
        self.device = device

    def generate_images(self, w: torch.Tensor):
        imgs = self.generator(w.to(self.device))
        return imgs

    @abstractmethod
    def run(self, target_label: int) -> AttackResult:
        pass

    def evaluate(self, result: AttackResult) -> AttackMetrics:
        pass