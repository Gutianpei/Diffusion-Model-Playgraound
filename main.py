import argparse
import traceback
import logging
import yaml
import sys
import os
import torch
import numpy as np

from ourddpm import OurDDPM
from configs.paths_config import HYBRID_MODEL_PATHS


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


