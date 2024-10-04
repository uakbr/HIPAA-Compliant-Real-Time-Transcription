# Custom secure memory allocator and erasure functions

import ctypes
import numpy as np

class SecureAllocator:
    def __init__(self):
        # Keep track of allocated buffers
        self.buffers = []

    def allocate(self, size):
        # Allocate a zero-initialized buffer
        buffer = np.zeros(size, dtype=np.uint8)
        self.buffers.append(buffer)
        return buffer

    def deallocate(self, buffer):
        # Overwrite the memory with zeros before deallocation
        self.zero_memory(buffer)
        if buffer in self.buffers:
            self.buffers.remove(buffer)
        del buffer

    def zero_memory(self, buffer):
        # Securely overwrite the buffer in-place with zeros
        ctypes.memset(buffer.ctypes.data, 0, buffer.nbytes)

    def cleanup(self):
        # Securely deallocate all buffers
        for buffer in self.buffers:
            self.deallocate(buffer)
        self.buffers.clear()