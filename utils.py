import torch
import os
import yaml



def save_experiment(model, seed, name, algorithm, net_structure, batch_size, 
                    decorrelation, lr_fwd, lr_decor, performances, correlation):

    # Create dictionary
    sim_params = {
        "algorithm": algorithm,
        "net_structure": net_structure,
        "batch_size": batch_size,
        "decorrelation": decorrelation,
        "seed": seed,
        "LR": {
            "fwd": lr_fwd,
            "decor": lr_decor,
        },
        "performances": performances,
        "correlation": correlation
    }

    # Create directory with summary of execution plus trained model
    if not os.path.exists('./experiments'):
        os.makedirs('./experiments')
    path = './experiments/' + str(name) + '_' + str(algorithm)
    if decorrelation:
        path += '_decor'
    path += '_batch'+str(batch_size)+'_LRs'+str(lr_fwd)+'-'+str(lr_decor)+'_seed'+str(seed)
    os.makedirs(path)

    # Saving configuration
    with open(path + '/sim_params.yml', 'w') as yaml_file:
        yaml.dump(sim_params, yaml_file, default_flow_style=False)

    # Save trained model
    torch.save(model.state_dict(), path+'/trained_model.pt')



def softmax():
    return lambda x: torch.exp(x) / torch.sum(torch.exp(x), dim=1, keepdim=True)


def leaky_relu(ALPHA):
    return lambda x: torch.where(x >= 0, x, ALPHA * x)


def relu():
    return lambda x: torch.where(x >= 0, x, 0)


def linear():
    return lambda x: x


def tanh():
    return lambda x: (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))


def epoch_profiling(model, x_train, y_train, criterion, lr_fwd, noise_std, algorithm, optimizer, decorrelation_prof, total_lenght_sequence):

    # Initial stuff
    outs = model.reset_states() # Reset before input presentation
    optimizer.zero_grad()
    pred = []

    if algorithm == 'BP':

        for t in range(total_lenght_sequence):

            # Input data point
            last_out, outs = model(x_train[t], outs)
            pred.append(last_out)

            # Compute decorelation updates (spatially)
            if decorrelation_prof:
                model.compute_update_params(lr_decor=0)

        # Apply decorrelation updates
        if decorrelation_prof:
            model.apply_update_params()

        # Compute loss
        loss = criterion(torch.stack(pred), y_train)

        # Update weights
        loss.backward()
        optimizer.step()

    #-- NP --#
    elif algorithm == 'NP':
        model.node_perturbation(x_train, y_train, loss=criterion, lr=lr_fwd, noise_std=noise_std, decorrelation_prof=decorrelation_prof)

    #-- WP --#
    elif algorithm == 'WP':
        model.weight_perturbation(x_train, y_train, loss=criterion, lr=lr_fwd, noise_std=noise_std, decorrelation_prof=decorrelation_prof)

    #-- ANP --#
    elif algorithm == 'ANP':
        model.node_perturbation_activity(x_train, y_train, loss=criterion, lr=lr_fwd, noise_std=noise_std, decorrelation_prof=decorrelation_prof)



