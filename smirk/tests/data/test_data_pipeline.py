import pytest
import torch
import json
from pathlib import Path
from omegaconf import OmegaConf

from smirk.data.dataset import QueryDataset, make_transform, build_query_dataset
from smirk.data.sampler import Sampler
from smirk.scripts.sample import validate_latent_config
from smirk.scripts.generate_blackbox_attack_data import run_merge
from smirk.genforce import my_get_GD


# Fixtures for simulating the expected file structures
@pytest.fixture
def mock_sample_dir(tmp_path):
    """
    Creates a simulated 'samples/model_name_...' directory
    """
    sample_dir = tmp_path / "samples" / "dummy_model"
    sample_dir.mkdir(parents=True)
    
    # Create 2 fake batches of images, 4 images each, size 3x32x32
    batch_size = 4
    for i in range(1, 3):
        img_tensor = torch.rand(batch_size, 3, 32, 32)
        latent_tensor = torch.randn(batch_size, 512)
        torch.save(img_tensor, sample_dir / f"sample_{i}_img.pt")
        torch.save(latent_tensor, sample_dir / f"sample_{i}_latent.pt")

    # Create dummy manifest
    manifest = {
        "model": "dummy",
        "batch_files": [
            {"iteration": 1, "image_file": "sample_1_img.pt", "latent_file": "sample_1_latent.pt"},
            {"iteration": 2, "image_file": "sample_2_img.pt", "latent_file": "sample_2_latent.pt"}
        ]
    }
    with open(sample_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)
        
    return sample_dir

@pytest.fixture
def mock_logits_path(tmp_path):
    """Creates a simulated all_logits.pt file."""
    logits_dir = tmp_path / "blackbox_attack_data"
    logits_dir.mkdir(parents=True)
    logits_path = logits_dir / "all_logits.pt"
    
    # 2 batches * 4 images = 8 total logits, 10 classes
    dummy_logits = torch.randn(8, 10)
    torch.save(dummy_logits, logits_path)
    return logits_path

@pytest.fixture
def dummy_transform():
    return make_transform(resolution=32, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


# Tests for smirk/data/dataset.py
class TestDataset:
    
    def test_query_dataset_indexing(self):
        """Verify QueryDataset correctly unpacks batches to return single items."""
        batch_size = 4
        # 2 batches of 4 images
        batches = [torch.rand(batch_size, 3, 32, 32) for _ in range(2)]
        logits = torch.randn(8, 10)
        
        # Identity transform for testing
        transform = lambda x: x 
        
        dataset = QueryDataset(batches, logits, transform)
        
        assert len(dataset) == 8
        
        # Check first item logic
        img, logit = dataset[0]
        assert torch.allclose(img, batches[0][0])
        assert torch.allclose(logit, logits[0])
        
        # Check an item in the second batch
        img, logit = dataset[5]
        assert torch.allclose(img, batches[1][1])
        assert torch.allclose(logit, logits[5])

    def test_make_transform(self):
        """Verify custom transform scales to 255 and normalizes correctly."""
        transform = make_transform(resolution=16, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
        # Input tensor strictly in [0, 1]
        test_img = torch.ones(3, 32, 32) * 0.5 
        out = transform(test_img)
        
        # Should be resized to 16x16, and scaled by 255
        assert out.shape == (3, 16, 16)
        assert torch.allclose(out, torch.ones(3, 16, 16) * 127.5)

    def test_build_query_dataset(self, mock_sample_dir, mock_logits_path, dummy_transform):
        """Verify the dataset builds correctly from the directory structure."""
        dataset = build_query_dataset(mock_sample_dir, mock_logits_path, dummy_transform, "cpu")
        
        assert len(dataset) == 8
        assert len(dataset.image_batches) == 2
        
    def test_build_query_dataset_missing_manifest(self, tmp_path, mock_logits_path, dummy_transform):
        """Verify error is raised if manifest is missing."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        with pytest.raises(FileNotFoundError, match="manifest.json not found"):
            build_query_dataset(empty_dir, mock_logits_path, dummy_transform, "cpu")



# Tests for smirk/data/sampler.py
class TestSampler:
    
    @pytest.fixture
    def mock_generator(self, sampler_config):
        generator, _ = my_get_GD.main(
            "cpu",
            "stylegan_celeba_partial256",
            sampler_config.batch_size,
            sampler_config.batch_size,
            use_w_space=sampler_config.latent_space.use_w_space,
            use_discri=False,
            repeat_w=sampler_config.latent_space.repeat_w,
            use_z_plus_space=sampler_config.latent_space.use_z_plus_space,
            trunc_psi=sampler_config.latent_space.trunc_psi,
            trunc_layers=sampler_config.latent_space.trunc_layers,
        )
        return generator

    @pytest.fixture
    def sampler_config(self):
        return OmegaConf.create({
            "model": {"genforce_model": "test_model"},
            "latent_space": {"trunc_psi": 0.5, "trunc_layers": 8},
            "size": 3, # 3 iterations
            "batch_size": 6,
            "latent_dim": 512,
            "latent_space": {
                "use_w_space": False,
                "repeat_w": True,
                "use_z_plus_space": False,
                "num_layers": 14,
                "trunc_psi": 0.7,
                "trunc_layers": 8
            }
        })

    def test_generate_samples(self, mock_generator, sampler_config, tmp_path):
        """Verify that samples and latents are generated and saved."""
        sampler = Sampler(torch.device("cpu"), mock_generator, sampler_config)
        
        out_dir = tmp_path / "sampler_out"
        out_dir.mkdir()
        
        sampler.generate_samples(out_dir)
        
        assert sampler.manifest_data["completed_iterations"] == 3
        assert len(list(out_dir.glob("*_img.pt"))) == 3
        assert len(list(out_dir.glob("*_latent.pt"))) == 3

    def test_merge_vectors(self, mock_generator, sampler_config, tmp_path):
        """Verify that multiple latent vectors merge correctly into all_ws.pt"""
        sampler = Sampler(torch.device("cpu"), mock_generator, sampler_config)
        out_dir = tmp_path / "sampler_out"
        out_dir.mkdir()
        
        # Give it some dummy files to merge
        for i in range(1, sampler_config.size+1):
            vector = torch.randn(sampler_config.batch_size, sampler_config.latent_dim)
            torch.save(vector, out_dir / f"sample_{i}_latent.pt")
            
        sampler.merge_vectors(out_dir)
        
        assert (out_dir / "all_ws.pt").exists()
        merged = torch.load(out_dir / "all_ws.pt")
        assert merged.shape[0] == sampler_config.batch_size * sampler_config.size

    def test_validate_latent_config(self, sampler_config):
        # Valid config
        validate_latent_config(sampler_config.latent_space)
        
        # Invalid config (use_z_plus_space requires use_w_space)
        cfg_invalid_1 = OmegaConf.create({"latent_space": {"use_z_plus_space": True, "use_w_space": False, "repeat_w": False}})
        with pytest.raises(AssertionError):
            validate_latent_config(cfg_invalid_1.latent_space)

        # Invalid config (use_z_plus_space requires repeat_w=False)
        cfg_invalid_2 = OmegaConf.create({"latent_space": {"use_z_plus_space": True, "use_w_space": True, "repeat_w": True}})
        with pytest.raises(AssertionError):
            validate_latent_config(cfg_invalid_2.latent_space)



# Tests for smirk/scripts/2-generate_blackbox_attack_data.py
class TestGenerateBlackboxDataScript:

    def test_run_merge(self, tmp_path):
        """Verify intermediate logits merge into a single tensor and can be cleaned up."""
        out_dir = tmp_path / "bb_data"
        out_dir.mkdir()
        
        # Create dummy intermediate files
        for i in range(1, 4):
            torch.save(torch.ones(2, 10) * i, out_dir / f"sample_{i}_img_logits.pt")
            
        # Run merge and tell it to delete intermediates
        run_merge(out_dir, remove=True)
        
        assert (out_dir / "all_logits.pt").exists()
        merged = torch.load(out_dir / "all_logits.pt")
        
        # 3 batches of 2 = 6 total items
        assert merged.shape == (6, 10)
        
        # Check values preserved properly
        assert merged[0][0] == 1.0  # From sample 1
        assert merged[-1][0] == 3.0 # From sample 3
        
        # Ensure intermediates were removed
        assert len(list(out_dir.glob("sample_*_img_logits.pt"))) == 0

    def test_run_merge_no_files(self, tmp_path):
        """Should raise a RuntimeError if there are no intermediate files to merge."""
        with pytest.raises(RuntimeError, match="No intermediate logit files found"):
            run_merge(tmp_path, remove=False)