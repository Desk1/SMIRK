# smirk/utils/files.py - file handling and logging utils

# API
# ---------------------------------
# get_path(relative)               # get path to relative location from project root
# create_folder(folder)            # create folder(s)

import os, sys
from pathlib import Path
from omegaconf import DictConfig

#################
# File handling #
#################

def get_path(relative_location: str) -> Path:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    return PROJECT_ROOT / relative_location

def get_sampling_directory(cfg: DictConfig) -> Path:
    dirname = (
        f"{cfg.output_dir}/"
        f"{cfg.model.genforce_model}_"
        f"{cfg.latent_space.trunc_psi}_"
        f"{cfg.latent_space.trunc_layers}_"
        f"{cfg.size}"
    )

    return Path(dirname)

def get_blackbox_attack_data_directory(cfg: DictConfig) -> Path:
    dirname = (
        f"{cfg.output_dir}/"
        f"{cfg.target_dataset}/"
        f"{cfg.arch_name}/"
        f"{cfg.sampling_dataset}/"
    )

# todo: rewrite
def create_folder(folder):
    if os.path.exists(folder):
        assert os.path.isdir(folder), 'it exists but is not a folder'
    else:
        os.makedirs(folder)

#############
#  Logging  #
#############

class Tee(object):
    # from https://github.com/MKariya1998/GMI-Attack/blob/master/Celeba/utils.py
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        if '...' not in data:
            self.file.write(data)
        self.stdout.write(data)
        self.flush()

    def flush(self):
        self.file.flush()