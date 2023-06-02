import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
from needle_tester import *
from torch_tester import *
def init_data(m,n,p):
    A = np.random.randn(m, n).astype(np.float32)
    B = np.random.randn(n, p).astype(np.float32)
    return A, B
def test_matmul(tester, m, n, p, A, B):
    data_func = lambda :tester._matmul_data(m, n, p)
    func = lambda :tester._matmul(A,B)
    tester.test_method(data_func, func, "matmul")
    return tester._matmul(A,B)

if __name__ == "__main__":
    MATMUL_DIMS = [(128, 128, 128), (64, 4096, 1024)]
    # needle_cpu_tester = NeedleCPUTester()
    torch_cpu_tester = TorchCPUTester()
    needle_m1_tester = NeedleM1Tester()
    torch_m1_tester = TorchM1Tester()
    # needle_cuda_tester = NeedleCUDATester()
    # torch_cuda_tester = TorchCUDATester()
    result_dict = {}
    for m, n, p in MATMUL_DIMS:
        A, B = init_data( m, n, p)
        # test_matmul(needle_cpu_tester, m, n, p)
        result_dict['torch_cpu_tester'] = test_matmul(torch_cpu_tester, m, n, p, A, B)
        result_dict['needle_m1_tester'] = test_matmul(needle_m1_tester, m, n, p, A, B)
        result_dict['torch_m1_tester'] = test_matmul(torch_m1_tester, m, n, p, A, B)
        # test_matmul(needle_cuda_tester, m, n, p)
        # test_matmul(torch_cuda_tester, m, n, p)
    res = True
    test_val = list(result_dict.values())[0]
    print(test_val)
    
    for ele in result_dict.keys():
        if not np.array_equal(test_val, result_dict[ele]):
            res = False
            break
    
    # printing result
    print("Are all values similar in dictionary? : " + str(res))
    
    


        


