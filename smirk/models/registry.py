# smirk/models/registry.py - central model registry

# Registry stores any relevant model data in a ModelSpec dataclass
# Each model backbone registers itself with @register_model decorator in its respective file
# Project code can query the registry via API defined here, avoids if/elif string comparison chains

"""
API
---------------------------------
register_model(name)             # decorator; registers a ModelSpec
get_spec(name)                   # returns the ModelSpec for *name*
get_model(name, device, ...)     # instantiates and returns an nn.Module
get_resolution(name)             # returns the input resolution
get_mean(name)                   # returns the normalisation mean list
get_std(name)                    # returns the normalisation std list
list_models()                    # sorted list of all registered names
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Union, Mapping, Any


#############
# ModelSpec #
#############

@dataclass
class ModelSpec:
    # string key identifier for model
    name: str

    # preferred model input resolution
    # allow tuple for non square inputs e.g SphereFace
    resolution: Union[int, tuple]

    # channel normalisation mean in [0,255] space, RGB order unless stated otherwise in backbone
    mean: list[float]

    # channel normalisation mean in [0,255] space
    std: list[float]

    # callable -> nn.Module
    # builds and returns the model
    loader: Callable[[], nn.Module]

    # optional  callable (spec, num_experts) -> nn.Module (default = None)
    # wraps the base model into the multi head expert variant ('_E' suffix) used for surrogate training
    expert_wrapper: Optional[Callable[["ModelSpec", int], nn.Module]] = None

    # optional path to load model weights from
    weights_path: Optional[Path] = None


##############
#  Registry  #
##############

REGISTRY: Dict[str, ModelSpec] = {}

def register_model(
    name: str, # only allow name as positional argument for clarity
    *,
    resolution: Union[int, tuple],
    mean: list[float],
    std: list[float],
    expert_wrapper: Optional[Callable] = None,
    weights_path: Optional[Path] = None,
    **kwargs
):
    if name in REGISTRY:
        raise ValueError(f"{name} is already registered")
    
    def decorator(loader):
        spec = ModelSpec(
            name=name,
            resolution=resolution,
            mean=mean,
            std=std,
            loader=loader,
            expert_wrapper=expert_wrapper,
            weights_path=weights_path
        )

        REGISTRY[name] = spec

        return loader
    
    return decorator
    
def get_spec(name: str):
    if len(REGISTRY) == 0:
        raise RuntimeError(
            f"Registry is empty or has not been loaded"
            f"Include 'import smirk.models' to initialise registry"
        )
    if name not in REGISTRY:
        registered = sorted(REGISTRY)
        raise KeyError(
            f"Model '{name}' is not registered. "
            f"Available models: {registered}"
        )
    return REGISTRY[name]
    

def get_model(
    name: str,
    device: Union[str, torch.device],
    load_weights_file: bool = True
):
    spec = get_spec(name)
    model = spec.loader()
    weights = get_weights(spec, device)

    if load_weights_file and weights is not None:
        model.load_state_dict(get_weights(spec, device))
    
    model = model.to(device)
    return model
 
 
def get_expert_model(
    name: str,
    num_experts: int,
    num_classification: int,
    device: Union[str, torch.device],
    load_weights: bool = True
):
    spec = get_spec(name)
    if spec.expert_wrapper is None:
        raise ValueError(
            f"{name}' does not have an expert_wrapper defined. \n"
            f"Define an expert wrapper at smirk/models/backbones"
            )
    model = spec.expert_wrapper(
        spec,
        num_experts,
        num_classification,
        device,
        load_weights
    )
    
    model = model.to(device)
    return model

def get_weights(
    spec: ModelSpec,
    device: Union[str, torch.device]
):
    state_dict = None

    if spec.weights_path is not None:
        if spec.weights_path.exists():
            state_dict = torch.load(spec.weights_path, map_location=device)
        else:
            raise FileNotFoundError(
                f"Model {spec.name} does not have a weights file at {spec.weights_path} \n"
                "Set a valid weights_path in smirk/models/backbones/"
            )
        
    return state_dict
 
 
def get_resolution(name: str) -> Union[int, tuple]:
    return get_spec(name).resolution
 
 
def get_mean(name: str) -> list[float]:
    return get_spec(name).mean
 
 
def get_std(name: str) -> list[float]:
    return get_spec(name).std
 
 
def list_models() -> list[str]:
    return sorted(REGISTRY.keys())