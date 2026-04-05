# smirk/utils/files.py - file handling and logging utils

# API
# ---------------------------------
# get_path(relative)               # get path to relative location from project root
# create_folder(folder)            # create folder(s)

import os, sys
from pathlib import Path

#################
# File handling #
#################

def get_path(relative_location: str):
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    return str(PROJECT_ROOT / relative_location)

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