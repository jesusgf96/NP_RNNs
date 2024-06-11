import torch
import torch.optim as optim
from models import *
from utils import *
from datasets import *
import torch.profiler

# Choose GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# Parameters copy dataset generation
batch_size = 1
delay_time = 10
lenght_sequence = 100
amplitude = 10
samples = 1
total_lenght_sequence = 2 * lenght_sequence + delay_time


# Generate data
x_train, y_train = generate_copy_data(delay=delay_time, lenght=lenght_sequence, amp=amplitude, batch_size=batch_size)
x_test, y_test = generate_copy_data(delay=delay_time, lenght=lenght_sequence, amp=amplitude, batch_size=batch_size)


# Move data to GPU
x_train = x_train.to(device).type(torch.float32)
y_train = y_train.to(device).type(torch.float32)
x_test = x_train.to(device).type(torch.float32)
y_test = y_train.to(device).type(torch.float32)


# Parameters RNN
# seed =  42  101 300  482  708 
seed = 42
net_structure = [10, 500, 10]
act_func = tanh()
act_func_out = linear()
epochs = 1
lr_fwd = 0.0001
noise_std = 10e-4
algorithm = 'BP' #'ANP' #'NP' #'WP'
decorrelation = True


# Input decorrelation
if net_structure[0] == 1:
    decor_input = False
else:
    decor_input = True


# Instantiate model
model = RNN(net_structure, batch_size, act_func, act_func_out, decor_input=decor_input, seed=42, device=device)
_ = model.to(device)


# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr_fwd) # Adam

# Check initial memory
initial_memory_gpu = torch.cuda.memory_allocated()

# Use PyTorch profiler for detailed profiling (optional)
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    epoch_profiling(model, x_train, y_train, criterion, lr_fwd, noise_std, algorithm, optimizer, decorrelation, total_lenght_sequence)
profiling_results = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
# print(profiling_results)


# Extract CUDA time total
execution_time_gpu = float(profiling_results.split("Self CUDA time total: ")[-1].split("ms")[0])

# Measure the final memory usage
final_memory_gpu = torch.cuda.memory_allocated()
memory_usage_gpu = final_memory_gpu - initial_memory_gpu

# Show results
print("----------------")
print("Algorithm -", algorithm)
print("Decorrelation -", decorrelation)
print("----------------")
print("execution_time_gpu:", execution_time_gpu, "ms")
print("memory_usage_gpu:", memory_usage_gpu, "bytes")
