import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
from basic_tester import *
from needle_tester import *
from torch_tester import *
# from IPython import embed

def test_matmul(tester, m, n, p, A, B):
    data_func = lambda :tester._matmul_data(m, n, p, A, B)
    func = lambda :tester._matmul()
    tester.test_method(data_func, func, "matmul")
    return tester._matmul()

if __name__ == "__main__":
    MATMUL_DIMS = [(128, 128, 128), (64, 4096, 1024)]
    tester_list = [
        NeedleCPUTester(),
        TorchCPUTester(),
        NeedleM1Tester(),
        TorchM1Tester(),
        # NeedleCUDATester(),
        # TorchCUDATester(),
    ]
    result_dict = dict([(tester.name, {}) for tester in tester_list])
    for m, n, p in MATMUL_DIMS:
        A, B = Tester.generate_random_matmul_data(m, n, p)
        for tester in tester_list:
            result_dict[tester.name]["matmul"] = test_matmul(tester, m, n, p, A, B)
        res = True
        base_key = list(result_dict.keys())[0]
        for ele in result_dict.keys():
            print(ele)
            if ele == "torch_m1_tester":
                if not np.allclose(result_dict[base_key]["matmul"].numpy(), result_dict[ele]["matmul"].cpu().numpy(), rtol=1e-3, atol=1e-3):
                    res = False
                    break
            elif not np.allclose(result_dict[base_key]["matmul"].numpy(), result_dict[ele]["matmul"].numpy(), rtol=1e-3, atol=1e-3):
                res = False
                break
        print("Are all values similar in dictionary? : " + str(res))
    
    


        


