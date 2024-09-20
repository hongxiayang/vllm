from collections import namedtuple
from typing import Any, Dict, List, Optional, Union

import torch
from torch.distributed import ProcessGroup


import torch.distributed as dist
import os

# from vllm.model_executor.parallel_utils import pynccl_utils
# from vllm.model_executor.parallel_utils.custom_all_reduce import (
#     custom_all_reduce)
# from vllm.model_executor.parallel_utils.parallel_state import (
#     get_tensor_model_parallel_group, get_tensor_model_parallel_rank,
#     get_tensor_model_parallel_world_size, is_pynccl_enabled_for_all_reduce)


# def broadcast_object_list(obj_list: List[Any],
#                           src: int = 0,
#                           group: Optional[ProcessGroup] = None):
#     """Broadcast the input object list."""
#     group = group or torch.distributed.group.WORLD
#     ranks = torch.distributed.get_process_group_ranks(group)
#     assert src in ranks, f"Invalid src rank ({src})"

#     # Bypass the function if we are using only 1 GPU.
#     world_size = torch.distributed.get_world_size(group=group)
#     if world_size == 1:
#         return obj_list
#     # Broadcast.
#     torch.distributed.broadcast_object_list(obj_list, src=src, group=group)
#     return obj_list

# from vllm/executor/torchrun_gpu_executor.py
#   def execute_model(self,
#                       seq_group_metadata_list: List[SequenceGroupMetadata],
#                       blocks_to_swap_in: Dict[int, int],
#                       blocks_to_swap_out: Dict[int, int],
#                       blocks_to_copy: Dict[int, List[int]]) -> SamplerOutput:
#         output = self.driver_worker.execute_model(
#             seq_group_metadata_list=seq_group_metadata_list,
#             blocks_to_swap_in=blocks_to_swap_in,
#             blocks_to_swap_out=blocks_to_swap_out,
#             blocks_to_copy=blocks_to_copy,
#         )
#         if self.is_driver_worker:
#             broadcast_object_list([output], src=0)
#         else:
#             res = [None]
#             broadcast_object_list(res, src=0)
#             output = res[0]
#         return output

def test_broadcast_obj_list():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    print(f"local rank is {local_rank}, rank={rank} world_size={world_size}")
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=world_size, #8,
        rank=rank, # this is optional
        init_method="env://",
    )
    # if dist.get_rank() == 0:
    #     # Assumes world_size of 3.
    #     objects = ["foo", 12, {1: 2}] # any picklable object
    # else:
    #     objects = [None, None, None]

    # # Assumes backend is not NCCL
    # #device = torch.device("cpu")
    # device = torch.device(f"cuda:{dist.get_rank()}")
    # dist.broadcast_object_list(objects, src=0, device=device)

    if dist.get_rank() == 0:
        # Assumes world_size of 3.
        objects = [12] # any picklable object
    else:
        objects = [None]

    # Assumes backend is not NCCL
    #device = torch.device("cpu")
    # device = torch.device(f"cuda:{dist.get_rank()}")
    # dist.broadcast_object_list(objects, src=0, device=device)
    dist.broadcast_object_list(objects, src=0)

    print(f"objects = {objects}")


if __name__ == "__main__":
    test_broadcast_obj_list()