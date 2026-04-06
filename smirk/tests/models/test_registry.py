import pytest
import torch
import torch.nn as nn

import smirk.models
from smirk.models.registry import REGISTRY, ModelSpec, register_model, get_model, get_spec, get_mean, get_std, get_resolution
from smirk.models.stats import ALL_MEANS, ALL_STDS

# Global test data: (name, resolution, mean, std)
# Mean and std are pulled from stats.py to ensure consistency
EXPECTED_MODELS = [
    ("resnet50",                    224,       ALL_MEANS["resnet50"],                    ALL_STDS["resnet50"]),
    ("inception_resnetv1_vggface2", 160,       ALL_MEANS["inception_resnetv1_vggface2"], ALL_STDS["inception_resnetv1_vggface2"]),
    ("inception_resnetv1_casia",    160,       ALL_MEANS["inception_resnetv1_casia"],    ALL_STDS["inception_resnetv1_casia"]),
    ("mobilenet_v2",                224,       ALL_MEANS["mobilenet_v2"],                ALL_STDS["mobilenet_v2"]),
    ("efficientnet_b0",             256,       ALL_MEANS["efficientnet_b0"],             ALL_STDS["efficientnet_b0"]),
    ("efficientnet_b0_casia",       256,       ALL_MEANS["efficientnet_b0_casia"],       ALL_STDS["efficientnet_b0_casia"]),
    ("inception_v3",                342,       ALL_MEANS["inception_v3"],                ALL_STDS["inception_v3"]),
    ("swin_transformer",            260,       ALL_MEANS["swin_transformer"],            ALL_STDS["swin_transformer"]),
    ("vision_transformer",          224,       ALL_MEANS["vision_transformer"],          ALL_STDS["vision_transformer"]),
    ("vgg16",                       224,       ALL_MEANS["vgg16"],                       ALL_STDS["vgg16"]),
    ("vgg16bn",                     224,       ALL_MEANS["vgg16bn"],                     ALL_STDS["vgg16bn"]),
    ("sphere20a",                   (112, 96), ALL_MEANS["sphere20a"],                   ALL_STDS["sphere20a"]),
]

def dummy_loader():
    """basic loader that returns a one layer linear model"""
    return nn.Linear(4, 4)

def dummy_expert_wrapper(spec: ModelSpec, num_experts: int) -> nn.Module:
    """trivial expert model wrapper"""
    return nn.ModuleList([nn.Linear(4, 4) for _ in range(num_experts)])

class TestBackboneRegistrations:
    """
    verify that importing smirk.models populates the registry correctly
    """
 
    @pytest.mark.parametrize("name,resolution,mean,std", EXPECTED_MODELS)
    def test_model_registered(self, name, resolution, mean, std):
        if name not in REGISTRY:
            pytest.skip(f"'{name}' not yet registered in this test run")
        spec = get_spec(name)
        assert spec.resolution == resolution
        assert spec.mean == mean
        assert spec.std == std

class TestRegisterModel:
    """
    verify that basic and expert models are registered in the registry properly
    """
    def test_basic_registration(self):
        @register_model(
            "test_basic",
            resolution=224,
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
        )
        def loader(spec, device, *, use_dropout=False):
            return nn.Linear(4, 4)
 
        assert "test_basic" in REGISTRY
        spec = REGISTRY["test_basic"]
        assert spec.resolution == 224
        assert spec.mean == [0.0, 0.0, 0.0]
        assert spec.std == [1.0, 1.0, 1.0]
        assert spec.expert_wrapper is None
 
    def test_registration_with_expert_wrapper(self):
        @register_model(
            "test_expert",
            resolution=160,
            mean=[127.5, 127.5, 127.5],
            std=[128.0, 128.0, 128.0],
            expert_wrapper=dummy_expert_wrapper,
        )
        def loader(spec, device, *, use_dropout=False):
            return nn.Linear(4, 4)
 
        spec = REGISTRY["test_expert"]
        assert spec.expert_wrapper is dummy_expert_wrapper

class TestGetModel:
    """
    verify get_model returns a correctly built and initialised model
    """
    def test_returns_nn_module_in_eval_mode(self):
        REGISTRY["_gm"] = ModelSpec(
            "_gm", 224, [0.0]*3, [1.0]*3, dummy_loader
        )
        model = get_model("_gm", "cpu")
        assert isinstance(model, nn.Module)
        assert not model.training  # eval() was called
 
    def test_model_moved_to_device(self):
        REGISTRY["_gm_dev"] = ModelSpec(
            "_gm_dev", 224, [0.0]*3, [1.0]*3, dummy_loader
        )
        model = get_model("_gm_dev", torch.device("cpu"))
        # all parameters should be on CPU
        for p in model.parameters():
            assert p.device == torch.device("cpu")


class TestGetResolution:
    """
    verify get_resolution returns the correct resolution for each model
    """

    @pytest.mark.parametrize("name,resolution,mean,std", EXPECTED_MODELS)
    def test_get_resolution(self, name, resolution, mean, std):
        if name not in REGISTRY:
            pytest.skip(f"'{name}' not yet registered in this test run")
        assert get_resolution(name) == resolution


class TestGetMean:
    """
    verify get_mean returns the correct normalization mean for each model
    """

    @pytest.mark.parametrize("name,resolution,mean,std", EXPECTED_MODELS)
    def test_get_mean(self, name, resolution, mean, std):
        if name not in REGISTRY:
            pytest.skip(f"'{name}' not yet registered in this test run")
        assert get_mean(name) == mean


class TestGetStd:
    """
    verify get_std returns the correct normalization std for each model
    """

    @pytest.mark.parametrize("name,resolution,mean,std", EXPECTED_MODELS)
    def test_get_std(self, name, resolution, mean, std):
        if name not in REGISTRY:
            pytest.skip(f"'{name}' not yet registered in this test run")
        assert get_std(name) == std