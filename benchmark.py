import time
import torch
from flash_muon import matmul_transpose

# Baseline version
def baseline(G):
    return G @ G.T

# Compiled version
compiled = torch.compile(baseline)

def benchmark_matmul_transpose():
    # Define dimensions to test
    dims = [1024, 2048, 4096, 8192]
    funcs = [matmul_transpose, baseline, compiled]
    loop = 8
    # Ensure we are on GPU
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
                # Call the baseline function
                func(tensor)
            end_event.record()
            torch.cuda.synchronize()  # Wait for the events to complete
            time_taken = start_event.elapsed_time(end_event)  # Time in milliseconds
            line += f'{time_taken:.2f}\t\t'

        print(line)

# Run the benchmark
if __name__ == "__main__":
    benchmark_matmul_transpose()
