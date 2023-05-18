#!/usr/bin/python3

gpu_params = {
    "SP":       13800,       # 12117.8885,  # GFLOPS
    "DRAM":     651,         # 562.4747,    # GB/s
    "L2":       1879.3369,   # GB/s
    "SHMEM":    3421.6682,   # GB/s
    "ADD":      11781.7527,  # GB/s
}

# for conv1, conv_vectorize_kernel
kernel_params = {
    "compute_ops": 59190018048, # Related to input size
    "dram_bytes":  1139965248,   # Related to Memory Access Pattern
    # "l2_bytes":    1751956736,  

    "mix_compute":      0.7369333324645029, # ratio of compute instructions
    "mix_ldst":         0.21730319360366995, # ratio of load/store instructions
    "opmix_efficiency": 1,      # since most op is add and mul, so we can estimate this as 1
    "actived_warps":    7.97,
}

# # for conv2, conv_vectorize_kernel
# kernel_params = {
#     "compute_ops": 14797504512, # Related to input size
#     "dram_bytes":  48730624,   # Related to Memory Access Pattern

#     "mix_compute":      0.7369333324645029, # ratio of compute instructions
#     "mix_ldst":         0.21730319360366995, # ratio of load/store instructions
#     "opmix_efficiency": 1,      # since most op is add and mul, so we can estimate this as 1
#     "actived_warps":    0.5,
# }

operation_ratio = kernel_params["compute_ops"] / kernel_params["dram_bytes"]
instr_throughput_factors = {
    'compute':   1, # for fp_32
    'ls':      gpu_params['SP']*0.5 / gpu_params['SHMEM'], # 0.5 means each MAD isntr is accounted as 2 ops
    # we using ADD to estimate all other instructions
    'other':   gpu_params['SP']*0.5 / gpu_params['ADD']
}
comp_thoughput = gpu_params["SP"]
cost_ld = kernel_params['mix_ldst']*instr_throughput_factors['ls']
cost_compute = kernel_params['mix_compute']*instr_throughput_factors["compute"]
cost_others = (1-kernel_params['mix_compute']-kernel_params['mix_ldst'])*instr_throughput_factors['other']
inst_efficiency = cost_compute/(cost_others+cost_compute+cost_ld)
overall_efficiency = inst_efficiency*kernel_params['opmix_efficiency']
occupancy_ratio = min(kernel_params["actived_warps"], 1)

adjusted_peak_compute = comp_thoughput * overall_efficiency * occupancy_ratio
adjusted_operation_ratio = adjusted_peak_compute/gpu_params['DRAM']
is_compute_bound = operation_ratio>adjusted_operation_ratio
estimated_comp_throughput = adjusted_peak_compute if is_compute_bound else operation_ratio*gpu_params['DRAM']
estimated_exec_time = kernel_params['compute_ops']/(estimated_comp_throughput*10**9)*10**3

print("estimate GFLOPS:%.2f, Time:%.2f" % (estimated_comp_throughput, estimated_exec_time))
# conv1 : predicted : 8461.06; real: 8914
# conv2 : predicted : 4230.53; real: 4425