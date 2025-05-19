import sys
import time
import torch
try:
    from flash_muon import matmul_transpose, fast_newtonschulz
except ImportError as e:
    print("Failed to import fast_newtonschulz from flash_muon. Please ensure the module is installed:")
    print("\tgit clone https://github.com/nil0x9/flash-muon.git && pip install -e ./")
    sys.exit(1)
try:
    import pandas as pd
except ImportError as e:
    print("This script requires pandas to run:\n\tpip install pandas")
    sys.exit(1)
pd.set_option('display.float_format',  '{:,.3f}'.format)
    
from collections import defaultdict

# Baseline version
def torch_matmul_transpose(G):
    return G @ G.T

def torch_zeropower_via_newtonschulz5(G, steps=5):
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    We opt to use a quintic iteration whose coefficients are selected to maximize the slope at zero.
    For the purpose of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X

def benchmark(name, baseline, impl, warmup=25, rep=100):
    # Define dimensions to test
    dims = [1024, 2048, 4096, 8192]
    compiled = torch.compile(baseline)
    funcs = [impl, baseline, compiled]
    # Ensure we are on GPU
    print(f"\nbenchmark {name}:")
    benchmark_result = defaultdict(list)
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for dim in dims:
        # Create a random tensor of shape [dim, dim]
        tensor = torch.randn(dim, dim, device='cuda').bfloat16()
        
        benchmark_result['device'].append(device_name)
        benchmark_result['dim'].append(dim)
        for idx, func in enumerate(funcs):
            torch.cuda.empty_cache()
            # warmup
            func(tensor)

            start_event.record()
            for _ in range(rep):
                # Call the function
                func(tensor)
            end_event.record()
            torch.cuda.synchronize()  # Wait for the events to complete
            time_taken = start_event.elapsed_time(end_event)/rep  # Time in milliseconds
            benchmark_result[['flash(ms)','torch(ms)','compiled(ms)'][idx]].append(time_taken)

    print(pd.DataFrame(benchmark_result))

# Run the benchmark
if __name__ == "__main__":
    benchmark(name='matmul transponse', baseline=torch_matmul_transpose, impl=matmul_transpose)
    benchmark(name='zeropower', baseline=torch_zeropower_via_newtonschulz5, impl=fast_newtonschulz)
