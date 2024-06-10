import torch
from training import *
from utils import *


# Choose GPU
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

# Parameters
# seed =  42  101 300  482  708 
seed = 42
dataset = 'copy' #'mackey-glass' #'weather' #'copy' #'psMNIST' 
algorithm = 'WP' #'NP' #'ANP' #'BP'
batch_size = 1 #10 #100
lr_fwd = 0.001
lr_decor = 1e-4 #1e-7
decorrelation = True
input_decorrelation = False
hidden_units = 500 #1000
epochs = 500
noise_std = 0.001 
act_func = tanh()
act_func_out = linear()
loss_name = 'mse' #'ce'
opt = 'adam' #'sgd'


# Define network architecture
if dataset == 'copy':
    net_structure = [10, hidden_units, 10]
elif dataset == 'psMNIST':
    net_structure = [1, hidden_units, 10]
elif dataset == 'weather':
    net_structure = [9, hidden_units, 1]
else:
    net_structure = [1, hidden_units, 1]


# Training loop
if dataset == 'mackey-glass':
    performances, correlation, model = training_mackey_glass(net_structure=net_structure, act_func=act_func, act_func_out=act_func_out, epochs=epochs, 
                                                             algorithm=algorithm, batch_size=batch_size, lr_fwd=lr_fwd, decorrelation=decorrelation, input_decorrelation=input_decorrelation,
                                                             lr_decor=lr_decor, noise_std=noise_std, loss_name=loss_name, opt=opt, device=device, seed=seed)
elif dataset == 'copy':
    performances, correlation, model = training_copy(net_structure=net_structure, act_func=act_func, act_func_out=act_func_out, epochs=epochs, 
                                                     algorithm=algorithm, batch_size=batch_size, lr_fwd=lr_fwd, decorrelation=decorrelation, input_decorrelation=input_decorrelation,
                                                     lr_decor=lr_decor, noise_std=noise_std, loss_name=loss_name, opt=opt, device=device, seed=seed)
elif dataset == 'psMNIST':
    performances, correlation, model = training_psMNIST(net_structure=net_structure, act_func=act_func, act_func_out=act_func_out, epochs=epochs, 
                                                        algorithm=algorithm, batch_size=batch_size, lr_fwd=lr_fwd, decorrelation=decorrelation, input_decorrelation=input_decorrelation,
                                                        lr_decor=lr_decor, noise_std=noise_std, loss_name=loss_name, opt=opt, device=device, seed=seed)
elif dataset == 'weather':
    performances, correlation, model = training_weather(net_structure=net_structure, act_func=act_func, act_func_out=act_func_out, epochs=epochs, 
                                                        algorithm=algorithm, batch_size=batch_size, lr_fwd=lr_fwd, decorrelation=decorrelation, input_decorrelation=input_decorrelation,
                                                        lr_decor=lr_decor, noise_std=noise_std, loss_name=loss_name, opt=opt, device=device, seed=seed)
else:
    print('Dataset not supported!')
    exit(0)
    

# Save log experiment and trained model
save_experiment(model=model, seed=seed, name=dataset, algorithm=algorithm, net_structure=net_structure, batch_size=batch_size,
                decorrelation=decorrelation, lr_fwd=lr_fwd, lr_decor=lr_decor, performances=performances, correlation=correlation)
