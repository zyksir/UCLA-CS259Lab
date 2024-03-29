import sys
import time
import numpy as np
from abc import ABC, abstractmethod

class Tester:
    def __init__(self) -> None:
        self.name = "BasicTester"
        self.num_repeats = 100
        pass

    @staticmethod
    def generate_random_matmul_data(m, n, p):
        A = np.random.randn(m, n).astype(np.float32)
        B = np.random.randn(n, p).astype(np.float32)
        return A, B

    @abstractmethod
    def _matmul_data(self, A, B):
        return
    
    @abstractmethod
    def _matmul(self):
        return
    
    @abstractmethod
    def _conv_data(self, Z_shape, W_shape, stride, padding):
        self._Z = np.random.randn(*Z_shape)*5
        self._Z = self._Z.astype(np.float32)
        self._W = np.random.randn(*W_shape)*5
        self._W = self._W.astype(np.float32)
        return 0

    @abstractmethod
    def _conv(self, padding, stride):
        return

    def test_method(self, data_func, func, func_name):
        print("**************************")
        print(f"start to test {func_name} in {self.name}")
        count = data_func()
        start = time.time()
        for _ in range(self.num_repeats):
            func()
        time_spent = time.time() - start
        gflops = count/time_spent*1e-9*self.num_repeats
        print(f"Time: {time_spent:.2f}s, GLOPS: {gflops:.2f}gflops")
        print("**************************")

     
