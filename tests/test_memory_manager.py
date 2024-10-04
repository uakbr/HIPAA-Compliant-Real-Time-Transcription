# Unit tests for secure memory allocation and deallocation

import unittest
from src.utils.memory_manager import SecureAllocator
import numpy as np

class TestMemoryManager(unittest.TestCase):
    def setUp(self):
        self.allocator = SecureAllocator()

    def test_allocate_and_deallocate(self):
        data = np.array([1, 2, 3], dtype=np.int16)
        buffer = self.allocator.allocate_buffer(data)
        self.assertTrue(np.array_equal(buffer, data))
        # Original data should be zeroed
        self.assertTrue(np.all(data == 0))
        self.allocator.deallocate(buffer)

    def test_cleanup(self):
        data = np.array([1, 2, 3], dtype=np.int16)
        buffer = self.allocator.allocate_buffer(data)
        self.allocator.cleanup()
        self.assertEqual(len(self.allocator.buffers), 0)

if __name__ == '__main__':
        unittest.main()