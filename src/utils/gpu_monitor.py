# Monitors GPU availability and usage for optimization

import torch

class GPUMonitor:
    def __init__(self):
        pass

    def is_gpu_available(self):
        return torch.cuda.is_available()

    def get_device_name(self):
        if self.is_gpu_available():
            return torch.cuda.get_device_name(0)
        else:
            return "CPU"

    def get_gpu_memory_info(self):
        if self.is_gpu_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            reserved_memory = torch.cuda.memory_reserved(0)
            allocated_memory = torch.cuda.memory_allocated(0)
            free_memory = total_memory - allocated_memory
            return {
                'total_memory': total_memory,
                'reserved_memory': reserved_memory,
                'allocated_memory': allocated_memory,
                'free_memory': free_memory,
            }
        else:
            return None