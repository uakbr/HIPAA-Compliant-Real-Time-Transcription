import unittest
from src.utils.memory_manager import MemoryPool, SecureAllocator
import numpy as np

class TestMemoryPool(unittest.TestCase):
    def setUp(self):
        self.pool_size = 1024 * 1024  # 1 MB
        self.memory_pool = MemoryPool(self.pool_size)

    def test_allocate_deallocate(self):
        buffer = self.memory_pool.allocate(1024)
        self.assertIsNotNone(buffer)
        self.memory_pool.deallocate(buffer)

    def test_memory_exhaustion(self):
        with self.assertRaises(MemoryError):
            self.memory_pool.allocate(self.pool_size + 1)

    def test_cleanup(self):
        buffer = self.memory_pool.allocate(1024)
        self.memory_pool.cleanup()
        self.assertEqual(self.memory_pool.offset, 0)
        self.assertEqual(len(self.memory_pool.allocations), 0)

class TestSecureAllocator(unittest.TestCase):
    def setUp(self):
        self.allocator = SecureAllocator(pool_size=1024 * 1024)

    def test_allocate_buffer(self):
        data = np.array([1, 2, 3], dtype=np.float32)
        buffer = self.allocator.allocate_buffer(data)
        self.assertTrue(np.array_equal(buffer, data))
        self.assertTrue(np.all(data == 0))  # Original data zeroed out

if __name__ == '__main__':
    unittest.main()