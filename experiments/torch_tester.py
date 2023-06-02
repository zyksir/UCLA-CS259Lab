from basic_tester import Tester

import torch

class TorchCPUTester(Tester):
    def __init__(self) -> None:
        super(TorchCPUTester, self).__init__()
        self.device = "cpu"
        self.name = "torch_cpu_tester"
    
    def _matmul_data(self, m, n, p):
        super(TorchCPUTester, self)._matmul_data(m, n, p)
        return m*n*p
    
    # def _matmul(self):
    #     self.A @ self.B
    def _matmul(self, A, B):
        self.A = torch.from_numpy(self._A).to(self.device)
        self.B = torch.from_numpy(self._B).to(self.device)
        return A @ B
    
    def _conv_data(self, Z_shape, W_shape, stride, padding):
        super(TorchCPUTester, self)._conv_data(Z_shape, W_shape, stride, padding)
        self.Z = torch.from_numpy(self._Z).to(self.device)
        self.W = torch.from_numpy(self._W).to(self.device)
    
    def _conv(self, padding, stride):
        out = torch.nn.functional.conv2d(self.Z.permute(0, 3, 1, 2), self.W.permute(3, 2, 0, 1).contiguous(), padding=padding, stride=stride)
        out2 = out.sum()
        out2.backward()


class TorchM1Tester(TorchCPUTester):
    def __init__(self) -> None:
        super(TorchM1Tester, self).__init__()
        self.device = "mps"
        self.name = "torch_m1_tester"

class TorchCUDATester(TorchCPUTester):
    def __init__(self) -> None:
        super(TorchCUDATester, self).__init__()
        self.device = "cuda"
        self.name = "torch_cuda_tester"