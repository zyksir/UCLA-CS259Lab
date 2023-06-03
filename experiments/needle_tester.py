from basic_tester import Tester
import needle as ndl
from needle import backend_ndarray as nd

class NeedleCPUTester(Tester):
    def __init__(self) -> None:
        super(NeedleCPUTester, self).__init__()
        self.device = nd.cpu()
        self.name = "needle_cpu_tester"
    
    def _matmul_data(self, m, n, p, A, B):
        self.A = ndl.Tensor(nd.array(A), device=self.device)
        self.B = ndl.Tensor(nd.array(B), device=self.device)
        return 2*m*n*p
    
    def _matmul(self):
        return self.A @ self.B
    
    def _conv_data(self, Z_shape, W_shape, stride, padding):
        super(NeedleCPUTester, self)._conv_data(Z_shape, W_shape, stride, padding)
        self.Z = ndl.Tensor(self._Z, device=self.device)
        self.W = ndl.Tensor(self._W, device=self.device)
        return 
    
    def _conv(self, padding, stride):
        y = ndl.conv(self.Z, self.W, padding=padding, stride=stride)
        y2 = y.sum()
        y2.backward()


class NeedleM1Tester(NeedleCPUTester):
    def __init__(self) -> None:
        super(NeedleM1Tester, self).__init__()
        self.device = nd.m1()
        self.name = "needle_m1_tester"

class NeedleCUDATester(NeedleCPUTester):
    def __init__(self) -> None:
        super(NeedleCUDATester, self).__init__()
        self.device = nd.cuda()
        self.name = "needle_cuda_tester"
