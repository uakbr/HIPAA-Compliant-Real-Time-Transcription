# Custom secure memory allocator and erasure functions

import ctypes
import numpy as np
import threading

class MemoryPool:
    def __init__(self, size):
        self.pool = ctypes.create_string_buffer(size)
        self.lock = threading.Lock()
        self.offset = 0
        self.size = size
        self.allocations = {}

    def allocate(self, size):
        with self.lock:
            if self.offset + size > self.size:
                raise MemoryError("Memory pool exhausted.")
            ptr = ctypes.addressof(self.pool) + self.offset
            buffer = (ctypes.c_char * size).from_address(ptr)
            self.allocations[ptr] = size
            self.offset += size
            return buffer

    def deallocate(self, buffer):
        with self.lock:
            ptr = ctypes.addressof(buffer)
            size = self.allocations.pop(ptr, None)
            if size:
                # Zero out the memory
                ctypes.memset(ptr, 0, size)

    def cleanup(self):
        with self.lock:
            # Zero out entire pool
            ctypes.memset(ctypes.addressof(self.pool), 0, self.size)
            self.allocations.clear()
            self.offset = 0

class SecureAllocator:
    def __init__(self, pool_size=1024 * 1024 * 100):  # 100 MB
        self.memory_pool = MemoryPool(pool_size)

    def allocate_buffer(self, data):
        size = data.nbytes
        buffer = self.memory_pool.allocate(size)
        ctypes.memmove(buffer, data.ctypes.data, size)
        self.zero_memory(data)  # Zero out original data
        return np.frombuffer((ctypes.c_char * size).from_address(ctypes.addressof(buffer)), dtype=data.dtype)

    def deallocate(self, buffer):
        self.memory_pool.deallocate(buffer)

    def zero_memory(self, data):
        ctypes.memset(data.ctypes.data, 0, data.nbytes)

    def cleanup(self):
        self.memory_pool.cleanup()