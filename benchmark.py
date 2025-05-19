import time
import torch
from flash_muon import matmul_transpose, fast_newtonschulz

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

def benchmark(name, baseline, impl):
    # Define dimensions to test
    dims = [1024, 2048, 4096, 8192]
    compiled = torch.compile(baseline)
    funcs = [impl, baseline, compiled]
    loop = 16
    # Ensure we are on GPU
    print(f"\nbenchmark {name}:")
    print("device\t\tdim\tflash(ms)\ttorch(ms)\tcompiled(ms)")
    device_name = torch.cuda.get_device_name(torch.cuda.current_device()).split(' ')[-1]
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for dim in dims:
        # Create a random tensor of shape [dim, dim]
        tensor = torch.randn(dim, dim, device='cuda').bfloat16()
        
        line = f'{device_name}\t{dim}\t'
        for func in funcs:
            torch.cuda.empty_cache()
            # warmup
            func(tensor)

            start_event.record()
            for _ in range(loop):
                # Call the function
                func(tensor)
            end_event.record()
            torch.cuda.synchronize()  # Wait for the events to complete
            time_taken = start_event.elapsed_time(end_event)  # Time in milliseconds
            line += f'{time_taken:.2f}\t\t'

        print(line)

# Run the benchmark
if __name__ == "__main__":
    benchmark(name='matmul transponse', baseline=torch_matmul_transpose, impl=matmul_transpose)
    benchmark(name='zeropower', baseline=torch_zeropower_via_newtonschulz5, impl=fast_newtonschulz)
