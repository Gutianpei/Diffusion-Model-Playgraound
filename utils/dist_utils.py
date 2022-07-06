import os
import sys
import tempfile
import torch
import torch.distributed as dist


def dist_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29000'

    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def dist_cleanup():
    dist.destroy_process_group()

