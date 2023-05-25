import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from needle_tester import *
from torch_tester import *

def test_matmul(tester, m, n, p):
    data_func = lambda :tester._matmul_data(m, n, p)
    func = lambda :tester._matmul()
    tester.test_method(data_func, func, "matmul")

if __name__ == "__main__":
    MATMUL_DIMS = [(128, 128, 128), (4096, 4096, 4096)]
    needle_cpu_tester = NeedleCPUTester()
    torch_cpu_tester = TorchCPUTester()
    needle_m1_tester = NeedleM1Tester()
    # torch_m1_tester = TorchM1Tester()
    # needle_cuda_tester = NeedleCUDATester()
    # torch_cuda_tester = TorchCUDATester()
    for m, n, p in MATMUL_DIMS:
        test_matmul(needle_cpu_tester, m, n, p)
        test_matmul(torch_cpu_tester, m, n, p)
        test_matmul(needle_m1_tester, m, n, p)
        # test_matmul(torch_m1_tester, m, n, p)
        # test_matmul(needle_cuda_tester, m, n, p)
        # test_matmul(torch_cuda_tester, m, n, p)
    
    


        


