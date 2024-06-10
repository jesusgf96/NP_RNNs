import copy
import torch.optim as optim
from tqdm import tqdm
from utils import *
from models import *
from datasets import *
import wandb


def training_mackey_glass(net_structure, act_func, act_func_out, epochs, algorithm, batch_size,
                          lr_fwd, decorrelation, input_decorrelation, lr_decor, noise_std, loss_name, opt, device, seed):
    
    # Generate dataset and move to GPU
    lenght_sequence = 5000
    (x_train, y_train), (x_test, y_test) = generate_mackey_glass_data(length=lenght_sequence, n_batches=batch_size)
    x_train = torch.tensor(x_train).to(device).type(torch.float32)
    x_test = torch.tensor(x_test).to(device).type(torch.float32)
    y_train = torch.tensor(y_train).to(device).type(torch.float32)
    y_test = torch.tensor(y_test).to(device).type(torch.float32)

    # Start a new wandb run to track this execution
    name = str(algorithm)
    if decorrelation:
        name += 'dec_batch'+str(batch_size)+'_LRs'+str(lr_fwd)+'_'+str(lr_decor)+'_std'+str(noise_std)
    else:
        name += '_batch'+str(batch_size)+'_LR'+str(lr_fwd)+'_std'+str(noise_std)
    name += '_units'+str(net_structure[1])+'_seed'+str(seed)
    wandb.init(
        project="MackeyGlass_RNN", name=name,
        config={
        "architecture": net_structure
        }
    )

    # Instantiate model
    model = RNN(net_structure, batch_size, act_func, act_func_out, decor_input=input_decorrelation, seed=seed, device=device)
    model_noisy = copy.deepcopy(model)
    _ = model.to(device)
    _ = model_noisy.to(device)

    # Define loss and optimizer
    if loss_name == 'mse':
        criterion = nn.MSELoss()
    elif loss_name == 'ce':
        criterion = nn.CrossEntropyLoss()

    # Define optimizer
    if opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr_fwd) # SGD
    elif opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr_fwd) # Adam
    
    # Loss and correlation history
    performances = {"test": {"acc": [], "loss": []}, "train": {"acc": [], "loss": []}}
    correlation ={"input": [], "hidden":[]}

    # Grad tracking on for BP
    if algorithm == 'BP':
        
        # Iterate epochs
        for e in range(epochs):

            print("\n Epoch", e + 1)

            ## --- Learning --- ##

            # Initial stuff
            activity_hist_0 = []
            activity_hist_1 = []
            running_loss = 0.0
            outs = model.reset_states() # Reset before input presentation
            outs_noisy = model_noisy.reset_states() # Reset before input presentation
            pred = []

            for t in tqdm(range(lenght_sequence)):

                # Reset optimizer
                optimizer.zero_grad()

                # Input data point
                last_out, outs_new = model(x_train[t], outs)
                act = [h.detach() for h in model.h]
                pred.append(last_out)

                # Store activities
                activity_hist_0.append(outs[0].cpu().numpy())
                activity_hist_1.append(outs[1].cpu().numpy())

                # Compute and apply decorelation updates online
                if decorrelation:
                    model.update_params(lr_decor=lr_decor)
                    model_noisy.decor_layers[1].weight.data  = copy.deepcopy(model.decor_layers[1].weight.data)

                # Update in every timestep

                #-- BP --#
                if algorithm == 'BP':
                    loss = criterion(last_out, y_train[t])
                    loss.backward()
                    optimizer.step()

                # Update hidden states
                outs = copy.deepcopy(outs_new)

                # Track loss
                running_loss += loss.item()


            ## --- Testing --- ##

            # Initial stuff
            test_running_loss = 0.0
            pred_test = []

            for t in tqdm(range(lenght_sequence)):

                # Input data point
                last_out, outs = model(x_test[t], outs)
                pred_test.append(last_out)

                # Compute loss
                test_loss = criterion(last_out, y_test[t])
                test_running_loss += test_loss.item()

            # Compute loss and acc
            loss = running_loss / len(x_train)
            test_loss = test_running_loss / len(x_test)
            performances["train"]["loss"].append(loss)
            performances["test"]["loss"].append(test_loss)
            print('Train loss', loss)
            print('Test loss:', test_loss)

            # Compute neural activities and cross-correlation
            activity_hist_0 = np.array(activity_hist_0)
            activity_hist_0 = np.reshape(activity_hist_0, (activity_hist_0.shape[0]*activity_hist_0.shape[1], activity_hist_0.shape[2]))
            activity_hist_1 = np.array(activity_hist_1)
            activity_hist_1 = np.reshape(activity_hist_1, (activity_hist_1.shape[0]*activity_hist_1.shape[1], activity_hist_1.shape[2]))
            cov = activity_hist_0.transpose() @ activity_hist_0
            offdiag_cov_0 = (1.0 - np.eye(len(cov)))*cov
            cov = activity_hist_1.transpose() @ activity_hist_1
            offdiag_cov_1 = (1.0 - np.eye(len(cov)))*cov

            # Store correlation layers
            correlation['input'].append(np.mean(offdiag_cov_0))
            correlation['hidden'].append(np.mean(offdiag_cov_1))
            print('Cross-correlation input and hidden:', np.mean(offdiag_cov_0), np.mean(offdiag_cov_1))

            # Track progress in wandb
            wandb.log({"loss train": loss,
                        "loss test": test_loss,
                        "decor input": np.mean(offdiag_cov_0), "decor hidden": np.mean(offdiag_cov_1)})

    # Grad tracking off for NP, WP and ANP
    else:
        
        # Iterate epochs
        with torch.no_grad():
            for e in range(epochs):

                print("\n Epoch", e + 1)

                ## --- Learning --- ##

                # Initial stuff
                activity_hist_0 = []
                activity_hist_1 = []
                running_loss = 0.0
                outs = model.reset_states() # Reset before input presentation
                outs_noisy = model_noisy.reset_states() # Reset before input presentation
                pred = []

                for t in tqdm(range(lenght_sequence)):

                    # Reset optimizer
                    optimizer.zero_grad()

                    # Input data point
                    last_out, outs_new = model(x_train[t], outs)
                    act = [h.detach() for h in model.h]

                    if algorithm == 'NP' or algorithm == 'ANP':
                        last_out_noisy, outs_noisy, noise = model_noisy(x_train[t], outs_noisy, noise_node=True, noise_std=noise_std)
                        act_noisy = [h.detach() for h in model_noisy.h]

                    elif algorithm == 'WP':
                        last_out_noisy, outs_noisy, noise = model_noisy(x_train[t], outs_noisy, noise_weight=True, noise_std=noise_std)
                        act_noisy = [h.detach() for h in model_noisy.h]
                    pred.append(last_out)

                    # Store activities
                    activity_hist_0.append(outs[0].cpu().numpy())
                    activity_hist_1.append(outs[1].cpu().numpy())

                    # Compute and apply decorelation updates online
                    if decorrelation:
                        model.update_params(lr_decor=lr_decor)
                        model_noisy.decor_layers[1].weight.data  = copy.deepcopy(model.decor_layers[1].weight.data)

                    # Update in every timestep

                    #-- NP --#
                    if algorithm == 'NP':
                        loss = criterion(last_out, y_train[t])
                        loss_noisy = criterion(last_out_noisy, y_train[t])
                        loss_error = loss_noisy.item() - loss.item()
                        dW0 = lr_fwd * ((1/noise_std) * (loss_error)) * noise[0].transpose(0, 1) @ outs_new[0]
                        model.fwd_layers[0].weight.data = model.fwd_layers[0].weight.data - dW0
                        model_noisy.fwd_layers[0].weight.data = copy.deepcopy(model.fwd_layers[0].weight.data)
                        dW1 = lr_fwd * ((1/noise_std) * (loss_error)) * noise[1].transpose(0, 1) @ outs_new[1]
                        model.fwd_layers[1].weight.data = model.fwd_layers[1].weight.data - dW1
                        model_noisy.fwd_layers[1].weight.data = copy.deepcopy(model.fwd_layers[1].weight.data)
                        if t > 0:
                            dR = lr_fwd * ((1/noise_std) * (loss_error)) * noise[0].transpose(0, 1) @ outs[1]
                            model.rec_layers[0].weight.data = model.rec_layers[0].weight.data - dR
                            model_noisy.rec_layers[0].weight.data = copy.deepcopy(model.rec_layers[0].weight.data)

                    #-- WP --#
                    if algorithm == 'WP':
                        loss = criterion(last_out, y_train[t])
                        loss_noisy = criterion(last_out_noisy, y_train[t])
                        loss_error = loss_noisy.item() - loss.item()
                        dW0 = lr_fwd * ((1/noise_std) * (loss_error)) * noise[0]
                        model.fwd_layers[0].weight.data = model.fwd_layers[0].weight.data - dW0
                        model_noisy.fwd_layers[0].weight.data = copy.deepcopy(model.fwd_layers[0].weight.data)
                        dW1 = lr_fwd * ((1/noise_std) * (loss_error)) * noise[2]
                        model.fwd_layers[1].weight.data = model.fwd_layers[1].weight.data - dW1
                        model_noisy.fwd_layers[1].weight.data = copy.deepcopy(model.fwd_layers[1].weight.data)
                        if t > 0:
                            dR = lr_fwd * ((1/noise_std) * (loss_error)) * noise[1]
                            model.rec_layers[0].weight.data = model.rec_layers[0].weight.data - dR
                            model_noisy.rec_layers[0].weight.data = copy.deepcopy(model.rec_layers[0].weight.data)

                    #-- ANP --#
                    if algorithm == 'ANP':
                        loss = criterion(last_out, y_train[t])
                        loss_noisy = criterion(last_out_noisy, y_train[t])
                        N = torch.tensor(np.sum(model.net_structure), device=device)
                        loss_error = loss_noisy.item() - loss.item()
                        act_diff1 = act_noisy[1] - act[1]
                        act_diff2 = act_noisy[2] - act[2]
                        dW0 = (1/batch_size)*(torch.sqrt(N) * loss_error * ((act_diff1) / (torch.norm(act_diff1, 'fro')**2))).transpose(0, 1) @ outs_new[0]
                        model.fwd_layers[0].weight.data = model.fwd_layers[0].weight.data - lr_fwd * dW0
                        model_noisy.fwd_layers[0].weight.data = copy.deepcopy(model.fwd_layers[0].weight.data)
                        dW1 = (1/batch_size)*(torch.sqrt(N) * loss_error * ((act_diff2) / (torch.norm(act_diff2, 'fro')**2))).transpose(0, 1) @ outs_new[1]
                        model.fwd_layers[1].weight.data = model.fwd_layers[1].weight.data - lr_fwd * dW1
                        model_noisy.fwd_layers[1].weight.data = copy.deepcopy(model.fwd_layers[1].weight.data)
                        if t > 0:
                            dR = (1/batch_size)*(torch.sqrt(N) * (loss_error) * ((act_diff1) / (torch.norm(act_diff1, 'fro')**2))).transpose(0, 1) @ outs[1]
                            model.rec_layers[0].weight.data = model.rec_layers[0].weight.data - lr_fwd * dR
                            model_noisy.rec_layers[0].weight.data = copy.deepcopy(model.rec_layers[0].weight.data)

                    # Update hidden states
                    outs = copy.deepcopy(outs_new)

                    # Track loss
                    running_loss += loss.item()


                ## --- Testing --- ##

                # Initial stuff
                test_running_loss = 0.0
                pred_test = []

                for t in tqdm(range(lenght_sequence)):

                    # Input data point
                    last_out, outs = model(x_test[t], outs)
                    pred_test.append(last_out)

                    # Compute loss
                    test_loss = criterion(last_out, y_test[t])
                    test_running_loss += test_loss.item()

                # Compute loss and acc
                loss = running_loss / len(x_train)
                test_loss = test_running_loss / len(x_test)
                performances["train"]["loss"].append(loss)
                performances["test"]["loss"].append(test_loss)
                print('Train loss', loss)
                print('Test loss:', test_loss)

                # Compute neural activities and cross-correlation
                activity_hist_0 = np.array(activity_hist_0)
                activity_hist_0 = np.reshape(activity_hist_0, (activity_hist_0.shape[0]*activity_hist_0.shape[1], activity_hist_0.shape[2]))
                activity_hist_1 = np.array(activity_hist_1)
                activity_hist_1 = np.reshape(activity_hist_1, (activity_hist_1.shape[0]*activity_hist_1.shape[1], activity_hist_1.shape[2]))
                cov = activity_hist_0.transpose() @ activity_hist_0
                offdiag_cov_0 = (1.0 - np.eye(len(cov)))*cov
                cov = activity_hist_1.transpose() @ activity_hist_1
                offdiag_cov_1 = (1.0 - np.eye(len(cov)))*cov

                # Store correlation layers
                correlation['input'].append(np.mean(offdiag_cov_0))
                correlation['hidden'].append(np.mean(offdiag_cov_1))
                print('Cross-correlation input and hidden:', np.mean(offdiag_cov_0), np.mean(offdiag_cov_1))

                # Track progress in wandb
                wandb.log({"loss train": loss,
                            "loss test": test_loss,
                            "decor input": np.mean(offdiag_cov_0), "decor hidden": np.mean(offdiag_cov_1)})

    # Close wandb connection
    wandb.finish()

    # Return logs experiments and model
    return performances, correlation, model




def training_copy(net_structure, act_func, act_func_out, epochs, algorithm, batch_size,
                  lr_fwd, decorrelation, input_decorrelation, lr_decor, noise_std, loss_name, opt, device, seed):
    
    # Generate dataset and move to GPU
    delay = 10
    lenght = 100 
    total_lenght_sequence = 2 * lenght + delay
    x_train, y_train = generate_copy_data(delay=delay, lenght=lenght)
    x_train = x_train.to(device).type(torch.float32)
    y_train = y_train.to(device).type(torch.float32)
    x_test, y_test = generate_copy_data(delay=delay, lenght=lenght)
    x_test = x_test.to(device).type(torch.float32)
    y_test = y_test.to(device).type(torch.float32)

    # Start a new wandb run to track this execution
    name = 'D'+str(delay)+'_L'+str(lenght)+'_'+str(algorithm)
    if decorrelation:
        name += 'dec_batch'+str(batch_size)+'_LRs'+str(lr_fwd)+'_'+str(lr_decor)+'_std'+str(noise_std)
    else:
        name += '_batch'+str(batch_size)+'_LR'+str(lr_fwd)+'_std'+str(noise_std)
    name += '_units'+str(net_structure[1])+'_seed'+str(seed)
    wandb.init(
        project="capacity_RNN", name=name,
        config={
        "architecture": net_structure
        }
    )

    # Instantiate model
    model = RNN(net_structure, batch_size, act_func, act_func_out, decor_input=input_decorrelation, seed=seed, device=device)
    _ = model.to(device)

    # Define loss and optimizer
    if loss_name == 'mse':
        criterion = nn.MSELoss()
    elif loss_name == 'ce':
        criterion = nn.CrossEntropyLoss()

    # Define optimizer
    if opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr_fwd) # SGD
    elif opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr_fwd) # Adam
    
    # Loss and correlation history
    performances = {"test": {"acc": [], "loss": [], "error_bits": []}, "train": {"acc": [], "loss": [], "error_bits": []}}
    correlation ={"input": [], "hidden":[]}

    # Iterate epochs
    for e in range(epochs):

        print("\n Epoch", e + 1)

        ## --- Learning --- ##

        # Initial stuff
        activity_hist_0 = []
        activity_hist_1 = []
        outs = model.reset_states() # Reset before input presentation
        optimizer.zero_grad()
        pred = []

        for t in tqdm(range(total_lenght_sequence)):

            # Input data point
            last_out, outs = model(x_train[t], outs)
            pred.append(last_out)

            # Store activities
            activity_hist_0.append(outs[0].cpu().numpy())
            activity_hist_1.append(outs[1].cpu().numpy())

            # Compute decorelation updates (spatially)
            if decorrelation:
                model.compute_update_params(lr_decor=lr_decor)

        # Apply decorrelation updates
        if decorrelation:
            model.apply_update_params()

        # Compute loss
        loss = criterion(torch.stack(pred), y_train)

        # No FWD learning in the first epoch
        if e > -1:

            #-- BP --#
            if algorithm == 'BP':
                loss.backward()
                optimizer.step()

            #-- NP --#
            if algorithm == 'NP':
                model.node_perturbation(x_train, y_train, loss=criterion, lr=lr_fwd, noise_std=noise_std)

            #-- WP --#
            if algorithm == 'WP':
                model.weight_perturbation(x_train, y_train, loss=criterion, lr=lr_fwd, noise_std=noise_std)

            #-- ANP --#
            if algorithm == 'ANP':
                model.node_perturbation_activity(x_train, y_train, loss=criterion, lr=lr_fwd, noise_std=noise_std)


        ## --- Testing --- ##

        # Initial stuff
        outs = model.reset_states() # Reset before input presentation
        pred_test = []

        for t in tqdm(range(total_lenght_sequence)):

            # Input data point
            last_out, outs = model(x_test[t], outs)
            pred_test.append(last_out)

        # Compute loss
        loss_test = criterion(torch.stack(pred_test), y_test)

        # Compute train loss and acc
        performances["train"]["loss"].append(loss.item())
        print('Train loss', loss.item())
        error_bits = torch.count_nonzero(torch.abs(y_train-torch.where(torch.stack(pred)<0.5, 0, 1)))
        print('Error bits train:', error_bits)
        performances["train"]["error_bits"].append(error_bits.detach().cpu().numpy())

        # Compute test loss and acc
        performances["test"]["loss"].append(loss_test.item())
        print('Test loss', loss_test.item())
        error_bits_test = torch.count_nonzero(torch.abs(y_test-torch.where(torch.stack(pred_test)<0.5, 0, 1)))
        print('Error bits test:', error_bits_test)
        performances["test"]["error_bits"].append(error_bits_test.detach().cpu().numpy())

        # Compute neural activities and cross-correlation
        activity_hist_0 = np.array(activity_hist_0)
        activity_hist_0 = np.reshape(activity_hist_0, (activity_hist_0.shape[0]*activity_hist_0.shape[1], activity_hist_0.shape[2]))
        activity_hist_1 = np.array(activity_hist_1)
        activity_hist_1 = np.reshape(activity_hist_1, (activity_hist_1.shape[0]*activity_hist_1.shape[1], activity_hist_1.shape[2]))
        cov = activity_hist_0.transpose() @ activity_hist_0
        offdiag_cov_0 = (1.0 - np.eye(len(cov)))*cov
        cov = activity_hist_1.transpose() @ activity_hist_1
        offdiag_cov_1 = (1.0 - np.eye(len(cov)))*cov

        # Store correlation layers
        correlation['input'].append(np.mean(offdiag_cov_0))
        correlation['hidden'].append(np.mean(offdiag_cov_1))
        print('Cross-correlation input and hidden:', np.mean(offdiag_cov_0), np.mean(offdiag_cov_1))

        # Save progress in weights and biases
        wandb.log({"loss train": loss.item(), "error bits": error_bits,
                   "loss test": loss_test.item(), "error bits test": error_bits_test,
                    "decor input": np.mean(offdiag_cov_0), "decor hidden": np.mean(offdiag_cov_1)})

    # Close wandb connection
    wandb.finish()

    # Return logs experiments and model
    return performances, correlation, model




def training_psMNIST(net_structure, act_func, act_func_out, epochs, algorithm, batch_size,
                     lr_fwd, decorrelation, input_decorrelation, lr_decor, noise_std, loss_name, opt, device, seed):
    
    # Generate dataset and move to GPU
    (x_train, y_train), (x_test, y_test) = generate_psMNIST_data()
    x_train = torch.tensor(x_train).to(device).type(torch.float32)
    x_test = torch.tensor(x_test).to(device).type(torch.float32)
    y_train = torch.tensor(y_train).to(device).type(torch.float32)
    y_test = torch.tensor(y_test).to(device).type(torch.float32)

    # # Take subset
    # x_train = x_train[:1000]
    # y_train = y_train[:1000]
    # x_test = x_test[:1000]
    # y_test = y_test[:1000]

    # Needed later
    n_samples_train = len(x_train)
    n_samples_test = len(x_test)

    # Reshape data to batch size
    if batch_size > 1:
        x_train = [torch.squeeze(torch.stack([x for x in x_train[i*batch_size:(i+1)*batch_size]], axis=2), dim=1) for i in range(int(len(x_train)/batch_size))]
        x_test = [torch.squeeze(torch.stack([x for x in x_test[i*batch_size:(i+1)*batch_size]], axis=2), dim=1) for i in range(int(len(x_test)/batch_size))]
        y_train = [torch.squeeze(torch.stack([x for x in y_train[i*batch_size:(i+1)*batch_size]], axis=1), dim=0) for i in range(int(len(y_train)/batch_size))]
        y_test = [torch.squeeze(torch.stack([x for x in y_test[i*batch_size:(i+1)*batch_size]], axis=1), dim=0) for i in range(int(len(y_test)/batch_size))]

    # Start a new wandb run to track this execution
    name = str(algorithm)
    if decorrelation:
        name += 'dec_batch'+str(batch_size)+'_LRs'+str(lr_fwd)+'_'+str(lr_decor)+'_std'+str(noise_std)
    else:
        name += '_batch'+str(batch_size)+'_LR'+str(lr_fwd)+'_std'+str(noise_std)
    name += '_units'+str(net_structure[1])+'_seed'+str(seed)
    wandb.init(
        project="psMNIST", name=name,
        config={
        "architecture": net_structure
        }
    )

    # Instantiate model
    model = RNN(net_structure, batch_size, act_func, act_func_out, decor_input=input_decorrelation, seed=seed, device=device)
    _ = model.to(device)

    # Define loss and optimizer
    if loss_name == 'mse':
        criterion = nn.MSELoss()
    elif loss_name == 'ce':
        criterion = nn.CrossEntropyLoss()

    # Define optimizer
    if opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr_fwd) # SGD
    elif opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr_fwd) # Adam
    
    # Loss and correlation history
    performances = {"test": {"acc": [], "loss": []}, "train": {"acc": [], "loss": []}}
    correlation ={"input": [], "hidden":[]}

    # Iterate epochs
    for e in range(epochs):

        print("\n Epoch", e + 1)

        ## --- Learning --- ##

        # Initial stuff
        activity_hist_0 = []
        activity_hist_1 = []
        running_loss = 0.0
        running_acc = []

        # Iterate samples
        for i in tqdm(range(len(x_train))):
            outs = model.reset_states() # Reset before input presentation
            optimizer.zero_grad()
            lenght_sequence = x_train[i].shape[0]

            # Average outputs
            x_hat_hist_0 = []
            x_hat_hist_1 = []

            # Iterate sequence
            for t in range(lenght_sequence):

                # Input presentation
                last_out, outs = model(x_train[i][t], outs)

                # Store activities
                x_hat_hist_0.append(outs[0].cpu().numpy())
                x_hat_hist_1.append(outs[1].cpu().numpy())

                # Compute decorelation updates (spatially)
                if decorrelation:
                    model.compute_update_params(lr_decor=lr_decor)

            # Store output over time of input and hidden states
            activity_hist_0.append(np.mean(np.array(x_hat_hist_0), axis=0))
            activity_hist_1.append(np.mean(np.array(x_hat_hist_1), axis=0))

            # Apply decorrelation updates
            if decorrelation:
                model.apply_update_params()

            # Compute loss
            loss = criterion(last_out, y_train[i])


            # No FWD learning in the first epoch
            if e > 0:

                #-- BP --#
                if algorithm == 'BP':
                    loss.backward()
                    optimizer.step()

                #-- NP --#
                if algorithm == 'NP':
                    model.node_perturbation(x_train[i], y_train[i], loss=criterion, lr=lr_fwd, noise_std=noise_std)

                #-- WP --#
                if algorithm == 'WP':
                    model.weight_perturbation(x_train[i], y_train[i], loss=criterion, lr=lr_fwd, noise_std=noise_std)

                #-- ANP --#
                if algorithm == 'ANP':
                    model.node_perturbation_activity(x_train[i], y_train[i], loss=criterion, lr=lr_fwd, noise_std=noise_std)

            running_loss += loss.item()
            running_acc.append((torch.argmax(last_out, dim=1) == y_train[i].argmax(dim=1)).long().sum().cpu())


        ## --- Testing --- ##

        # Initial stuff
        test_running_loss = 0.0
        test_running_acc = []

        # Iterate samples
        for i in tqdm(range(len(x_test))):
            outs = model.reset_states() # Reset before input presentation
            lenght_sequence = x_test[i].shape[0]

            # Iterate sequence
            for t in range(lenght_sequence):

                # Input presentation
                last_out, outs = model(x_test[i][t], outs)

            # Compute loss
            loss = criterion(last_out, y_test[i])
            test_running_loss += loss.item()
            test_running_acc.append((torch.argmax(last_out, dim=1) == y_test[i].argmax(dim=1)).long().sum().cpu())


        # Compute loss and acc
        loss = running_loss / len(x_train)
        acc = np.sum(running_acc)/n_samples_train 
        test_loss = test_running_loss / len(x_test)
        test_acc = np.sum(test_running_acc)/n_samples_test 
        performances["train"]["loss"].append(loss)
        performances["train"]["acc"].append(float(acc))
        performances["test"]["loss"].append(test_loss)
        performances["test"]["acc"].append(float(test_acc))
        print('Train acc and loss', acc, loss)
        print('Test acc and loss:', test_acc, test_loss)

        # Compute neural activities and cross-correlation
        activity_hist_0 = np.array(activity_hist_0)
        activity_hist_0 = np.reshape(activity_hist_0, (activity_hist_0.shape[0]*activity_hist_0.shape[1], activity_hist_0.shape[2]))
        activity_hist_1 = np.array(activity_hist_1)
        activity_hist_1 = np.reshape(activity_hist_1, (activity_hist_1.shape[0]*activity_hist_1.shape[1], activity_hist_1.shape[2]))
        cov = activity_hist_0.transpose() @ activity_hist_0
        offdiag_cov_0 = (1.0 - np.eye(len(cov)))*cov
        cov = activity_hist_1.transpose() @ activity_hist_1
        offdiag_cov_1 = (1.0 - np.eye(len(cov)))*cov

        # Store correlation layers
        correlation['input'].append(np.mean(offdiag_cov_0))
        correlation['hidden'].append(np.mean(offdiag_cov_1))
        print('Cross-correlation input and hidden:', np.mean(offdiag_cov_0), np.mean(offdiag_cov_1))

        # Save progress in weights and biases
        wandb.log({"acc train": acc, "loss train": loss,
                    "acc test": test_acc, "loss test": test_loss,
                    "decor input": np.mean(offdiag_cov_0), "decor hidden": np.mean(offdiag_cov_1)})

    # Close wandb connection
    wandb.finish()

    # Return logs experiments and model
    return performances, correlation, model



def training_weather(net_structure, act_func, act_func_out, epochs, algorithm, batch_size,
                     lr_fwd, decorrelation, input_decorrelation, lr_decor, noise_std, loss_name, opt, device, seed):

    # Generate dataset and move to GPU
    # delay_pred = 1 # 24 # 48 
    delay_pred = 1 # in hours
    split_indx = 33600
    (x_train, y_train), (x_test, y_test) = generate_weather_data(delay_pred=delay_pred, batch_size=batch_size, split_indx=split_indx)
    lenght_sequence_train = len(x_train)
    lenght_sequence_test = len(x_test)
    x_train = torch.tensor(x_train).to(device).type(torch.float32)
    x_test = torch.tensor(x_test).to(device).type(torch.float32)
    y_train = torch.tensor(y_train).to(device).type(torch.float32)
    y_test = torch.tensor(y_test).to(device).type(torch.float32)

    # Start a new wandb run to track this execution
    name = str(delay_pred)+'h_'+str(algorithm)
    if decorrelation:
        if input_decorrelation:
            name += 'decin_batch'+str(batch_size)+'_LRs'+str(lr_fwd)+'_'+str(lr_decor)+'_std'+str(noise_std)
        else:
            name += 'dec_batch'+str(batch_size)+'_LRs'+str(lr_fwd)+'_'+str(lr_decor)+'_std'+str(noise_std)
    else:
        name += '_batch'+str(batch_size)+'_LR'+str(lr_fwd)+'_std'+str(noise_std)
    name += '_units'+str(net_structure[1])+'_seed'+str(seed)
    wandb.init(
        project="weather_RNN", name=name,
        config={
        "architecture": net_structure
        }
    )

    # Instantiate model
    model = RNN(net_structure, batch_size, act_func, act_func_out, decor_input=input_decorrelation, seed=seed, device=device)
    model_noisy = copy.deepcopy(model)
    _ = model.to(device)
    _ = model_noisy.to(device)

    # Define loss and optimizer
    if loss_name == 'mse':
        criterion = nn.MSELoss()
    elif loss_name == 'ce':
        criterion = nn.CrossEntropyLoss()

    # Define optimizer
    if opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr_fwd) # SGD
    elif opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr_fwd) # Adam
    
    # Loss and correlation history
    performances = {"test": {"acc": [], "loss": []}, "train": {"acc": [], "loss": []}}
    correlation ={"input": [], "hidden":[]}

    # Grad tracking on for BP
    if algorithm == 'BP':
        
        # Iterate epochs
        for e in range(epochs):

            print("\n Epoch", e + 1)

            ## --- Learning --- ##

            # Initial stuff
            activity_hist_0 = []
            activity_hist_1 = []
            # running_loss = 0.0
            
            # outs = model.reset_states() # Reset before input presentation
            outs_noisy = model_noisy.reset_states() # Reset before input presentation
            pred = []

            for t in tqdm(range(lenght_sequence_train)):

                # Reset optimizer
                optimizer.zero_grad()

                # Input data point
                last_out, outs_new = model(x_train[t], outs)
                act = [h.detach() for h in model.h]
                pred.append(last_out)

                # Store activities
                activity_hist_0.append(outs[0].cpu().numpy())
                activity_hist_1.append(outs[1].cpu().numpy())

                # Compute and apply decorelation updates online
                if decorrelation:
                    model.update_params(lr_decor=lr_decor)
                    model_noisy.decor_layers[1].weight.data  = copy.deepcopy(model.decor_layers[1].weight.data)

                # Update in every timestep

                #-- BP --#
                if algorithm == 'BP':
                    loss = criterion(last_out, y_train[t])
                    loss.backward()
                    optimizer.step()

                # Update hidden states
                outs = copy.deepcopy(outs_new)

                # Track loss
                running_loss += loss.item()


            ## --- Testing --- ##

            # Initial stuff
            test_running_loss = 0.0
            pred_test = []

            for t in tqdm(range(lenght_sequence_test)):

                # Input data point
                last_out, outs = model(x_test[t], outs)
                pred_test.append(last_out)

                # Compute loss
                test_loss = criterion(last_out, y_test[t])
                test_running_loss += test_loss.item()

            # Compute loss and acc
            loss = running_loss / len(x_train)
            test_loss = test_running_loss / len(x_test)
            performances["train"]["loss"].append(loss)
            performances["test"]["loss"].append(test_loss)
            print('Train loss', loss)
            print('Test loss:', test_loss)

            # Compute neural activities and cross-correlation
            activity_hist_0 = np.array(activity_hist_0)
            activity_hist_0 = np.reshape(activity_hist_0, (activity_hist_0.shape[0]*activity_hist_0.shape[1], activity_hist_0.shape[2]))
            activity_hist_1 = np.array(activity_hist_1)
            activity_hist_1 = np.reshape(activity_hist_1, (activity_hist_1.shape[0]*activity_hist_1.shape[1], activity_hist_1.shape[2]))
            cov = activity_hist_0.transpose() @ activity_hist_0
            offdiag_cov_0 = (1.0 - np.eye(len(cov)))*cov
            cov = activity_hist_1.transpose() @ activity_hist_1
            offdiag_cov_1 = (1.0 - np.eye(len(cov)))*cov

            # Store correlation layers
            correlation['input'].append(np.mean(offdiag_cov_0))
            correlation['hidden'].append(np.mean(offdiag_cov_1))
            print('Cross-correlation input and hidden:', np.mean(offdiag_cov_0), np.mean(offdiag_cov_1))

            # Track progress in wandb
            wandb.log({"loss train": loss,
                        "loss test": test_loss,
                        "decor input": np.mean(offdiag_cov_0), "decor hidden": np.mean(offdiag_cov_1)})

    # Grad tracking off for NP, WP and ANP
    else:
        
        # Iterate epochs
        with torch.no_grad():
            for e in range(epochs):

                print("\n Epoch", e + 1)

                ## --- Learning --- ##

                # Initial stuff
                activity_hist_0 = []
                activity_hist_1 = []
                running_loss = 0.0
                outs = model.reset_states() # Reset before input presentation
                outs_noisy = model_noisy.reset_states() # Reset before input presentation
                pred = []

                for t in tqdm(range(lenght_sequence_train)):

                    # Reset optimizer
                    optimizer.zero_grad()

                    # Input data point
                    last_out, outs_new = model(x_train[t], outs)
                    act = [h.detach() for h in model.h]

                    if algorithm == 'NP' or algorithm == 'ANP':
                        last_out_noisy, outs_noisy, noise = model_noisy(x_train[t], outs_noisy, noise_node=True, noise_std=noise_std)
                        act_noisy = [h.detach() for h in model_noisy.h]

                    elif algorithm == 'WP':
                        last_out_noisy, outs_noisy, noise = model_noisy(x_train[t], outs_noisy, noise_weight=True, noise_std=noise_std)
                        act_noisy = [h.detach() for h in model_noisy.h]
                    pred.append(last_out)

                    # Store activities
                    activity_hist_0.append(outs[0].cpu().numpy())
                    activity_hist_1.append(outs[1].cpu().numpy())

                    # Compute and apply decorelation updates online
                    if decorrelation:
                        model.update_params(lr_decor=lr_decor)
                        model_noisy.decor_layers[1].weight.data  = copy.deepcopy(model.decor_layers[1].weight.data)

                    # Update in every timestep

                    #-- NP --#
                    if algorithm == 'NP':
                        loss = criterion(last_out, y_train[t])
                        loss_noisy = criterion(last_out_noisy, y_train[t])
                        loss_error = loss_noisy.item() - loss.item()
                        dW0 = lr_fwd * ((1/noise_std) * (loss_error)) * noise[0].transpose(0, 1) @ outs_new[0]
                        model.fwd_layers[0].weight.data = model.fwd_layers[0].weight.data - dW0
                        model_noisy.fwd_layers[0].weight.data = copy.deepcopy(model.fwd_layers[0].weight.data)
                        dW1 = lr_fwd * ((1/noise_std) * (loss_error)) * noise[1].transpose(0, 1) @ outs_new[1]
                        model.fwd_layers[1].weight.data = model.fwd_layers[1].weight.data - dW1
                        model_noisy.fwd_layers[1].weight.data = copy.deepcopy(model.fwd_layers[1].weight.data)
                        if t > 0:
                            dR = lr_fwd * ((1/noise_std) * (loss_error)) * noise[0].transpose(0, 1) @ outs[1]
                            model.rec_layers[0].weight.data = model.rec_layers[0].weight.data - dR
                            model_noisy.rec_layers[0].weight.data = copy.deepcopy(model.rec_layers[0].weight.data)

                    #-- WP --#
                    if algorithm == 'WP':
                        loss = criterion(last_out, y_train[t])
                        loss_noisy = criterion(last_out_noisy, y_train[t])
                        loss_error = loss_noisy.item() - loss.item()
                        dW0 = lr_fwd * ((1/noise_std) * (loss_error)) * noise[0]
                        model.fwd_layers[0].weight.data = model.fwd_layers[0].weight.data - dW0
                        model_noisy.fwd_layers[0].weight.data = copy.deepcopy(model.fwd_layers[0].weight.data)
                        dW1 = lr_fwd * ((1/noise_std) * (loss_error)) * noise[2]
                        model.fwd_layers[1].weight.data = model.fwd_layers[1].weight.data - dW1
                        model_noisy.fwd_layers[1].weight.data = copy.deepcopy(model.fwd_layers[1].weight.data)
                        if t > 0:
                            dR = lr_fwd * ((1/noise_std) * (loss_error)) * noise[1]
                            model.rec_layers[0].weight.data = model.rec_layers[0].weight.data - dR
                            model_noisy.rec_layers[0].weight.data = copy.deepcopy(model.rec_layers[0].weight.data)

                    #-- ANP --#
                    if algorithm == 'ANP':
                        loss = criterion(last_out, y_train[t])
                        loss_noisy = criterion(last_out_noisy, y_train[t])
                        N = torch.tensor(np.sum(model.net_structure), device=device)
                        loss_error = loss_noisy.item() - loss.item()
                        act_diff1 = act_noisy[1] - act[1]
                        act_diff2 = act_noisy[2] - act[2]
                        dW0 = (1/batch_size)*(torch.sqrt(N) * loss_error * ((act_diff1) / (torch.norm(act_diff1, 'fro')**2))).transpose(0, 1) @ outs_new[0]
                        model.fwd_layers[0].weight.data = model.fwd_layers[0].weight.data - lr_fwd * dW0
                        model_noisy.fwd_layers[0].weight.data = copy.deepcopy(model.fwd_layers[0].weight.data)
                        dW1 = (1/batch_size)*(torch.sqrt(N) * loss_error * ((act_diff2) / (torch.norm(act_diff2, 'fro')**2))).transpose(0, 1) @ outs_new[1]
                        model.fwd_layers[1].weight.data = model.fwd_layers[1].weight.data - lr_fwd * dW1
                        model_noisy.fwd_layers[1].weight.data = copy.deepcopy(model.fwd_layers[1].weight.data)
                        if t > 0:
                            dR = (1/batch_size)*(torch.sqrt(N) * (loss_error) * ((act_diff1) / (torch.norm(act_diff1, 'fro')**2))).transpose(0, 1) @ outs[1]
                            model.rec_layers[0].weight.data = model.rec_layers[0].weight.data - lr_fwd * dR
                            model_noisy.rec_layers[0].weight.data = copy.deepcopy(model.rec_layers[0].weight.data)

                    # Update hidden states
                    outs = copy.deepcopy(outs_new)

                    # Track loss
                    running_loss += loss.item()


                ## --- Testing --- ##

                # Initial stuff
                test_running_loss = 0.0
                pred_test = []

                for t in tqdm(range(lenght_sequence_test)):

                    # Input data point
                    last_out, outs = model(x_test[t], outs)
                    pred_test.append(last_out)

                    # Compute loss
                    test_loss = criterion(last_out, y_test[t])
                    test_running_loss += test_loss.item()

                # Compute loss and acc
                loss = running_loss / len(x_train)
                test_loss = test_running_loss / len(x_test)
                performances["train"]["loss"].append(loss)
                performances["test"]["loss"].append(test_loss)
                print('Train loss', loss)
                print('Test loss:', test_loss)

                # Compute neural activities and cross-correlation
                activity_hist_0 = np.array(activity_hist_0)
                activity_hist_0 = np.reshape(activity_hist_0, (activity_hist_0.shape[0]*activity_hist_0.shape[1], activity_hist_0.shape[2]))
                activity_hist_1 = np.array(activity_hist_1)
                activity_hist_1 = np.reshape(activity_hist_1, (activity_hist_1.shape[0]*activity_hist_1.shape[1], activity_hist_1.shape[2]))
                cov = activity_hist_0.transpose() @ activity_hist_0
                offdiag_cov_0 = (1.0 - np.eye(len(cov)))*cov
                cov = activity_hist_1.transpose() @ activity_hist_1
                offdiag_cov_1 = (1.0 - np.eye(len(cov)))*cov

                # Store correlation layers
                correlation['input'].append(np.mean(offdiag_cov_0))
                correlation['hidden'].append(np.mean(offdiag_cov_1))
                print('Cross-correlation input and hidden:', np.mean(offdiag_cov_0), np.mean(offdiag_cov_1))

                # Squared lower triangular matrix correlation
                sq_corr_hidden = offdiag_cov_1[np.tril_indices(len(offdiag_cov_1), -1)] ** 2
                

                # Track progress in wandb
                wandb.log({"loss train": loss,
                            "loss test": test_loss,
                            "decor input": np.mean(offdiag_cov_0), "decor hidden": np.mean(offdiag_cov_1),
                            "sq tril corr mean": np.mean(sq_corr_hidden), "sq tril corr max": np.max(sq_corr_hidden), "sq tril corr min": np.min(sq_corr_hidden),
                            "sq tril corr median": np.median(sq_corr_hidden), "sq tril corr Q1": np.percentile(sq_corr_hidden, 25), "sq tril corr Q3": np.percentile(sq_corr_hidden, 75)})

    # Close wandb connection
    wandb.finish()

    # Return logs experiments and model
    return performances, correlation, model



# ### MARCEL'S DERIVATION
def training_weather(net_structure, act_func, act_func_out, epochs, algorithm, batch_size,
                     lr_fwd, decorrelation, input_decorrelation, lr_decor, noise_std, loss_name, opt, device, seed):

    # Generate dataset and move to GPU
    # delay_pred = 1 # 24 # 48 
    delay_pred = 1 # in hours
    split_indx = 33600
    (x_train, y_train), (x_test, y_test) = generate_weather_data(delay_pred=delay_pred, batch_size=batch_size, split_indx=split_indx)
    lenght_sequence_train = len(x_train)
    lenght_sequence_test = len(x_test)
    x_train = torch.tensor(x_train).to(device).type(torch.float32)
    x_test = torch.tensor(x_test).to(device).type(torch.float32)
    y_train = torch.tensor(y_train).to(device).type(torch.float32)
    y_test = torch.tensor(y_test).to(device).type(torch.float32)

    # Start a new wandb run to track this execution
    name = str(delay_pred)+'h_'+str(algorithm)
    if decorrelation:
        if input_decorrelation:
            name += 'decin_batch'+str(batch_size)+'_LRs'+str(lr_fwd)+'_'+str(lr_decor)+'_std'+str(noise_std)
        else:
            name += 'dec_batch'+str(batch_size)+'_LRs'+str(lr_fwd)+'_'+str(lr_decor)+'_std'+str(noise_std)
    else:
        name += '_batch'+str(batch_size)+'_LR'+str(lr_fwd)+'_std'+str(noise_std)
    name += '_units'+str(net_structure[1])+'_seed'+str(seed)
    wandb.init(
        project="weather_RNN", name=name,
        config={
        "architecture": net_structure
        }
    )

    # Instantiate model
    model = RNN(net_structure, batch_size, act_func, act_func_out, decor_input=input_decorrelation, seed=seed, device=device)
    model_noisy = copy.deepcopy(model)
    _ = model.to(device)
    _ = model_noisy.to(device)

    # Define loss and optimizer
    if loss_name == 'mse':
        criterion = nn.MSELoss()
    elif loss_name == 'ce':
        criterion = nn.CrossEntropyLoss()

    # Define optimizer
    if opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr_fwd) # SGD
    elif opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr_fwd) # Adam
    
    # Loss and correlation history
    performances = {"test": {"acc": [], "loss": []}, "train": {"acc": [], "loss": []}}
    correlation ={"input": [], "hidden":[]}

    with torch.no_grad():
        for e in range(epochs):

            print("\n Epoch", e + 1)

            ## --- Learning --- ##

            # Initial stuff
            activity_hist_0 = []
            activity_hist_1 = []
            running_loss = 0.0
            outs = model.reset_states() # Reset before input presentation
            outs_noisy = model_noisy.reset_states() # Reset before input presentation
            pred = []
            losses_diff = []
            et_w0 = []
            et_w1 = []
            et_r = []

            for t in tqdm(range(lenght_sequence_train)):

                # Reset optimizer
                optimizer.zero_grad()

                # Input data point
                last_out, outs_new = model(x_train[t], outs)
                act = [h.detach() for h in model.h]

                if algorithm == 'NP' or algorithm == 'ANP':
                    last_out_noisy, outs_noisy, noise = model_noisy(x_train[t], outs_noisy, noise_node=True, noise_std=noise_std)
                    act_noisy = [h.detach() for h in model_noisy.h]

                elif algorithm == 'WP':
                    last_out_noisy, outs_noisy, noise = model_noisy(x_train[t], outs_noisy, noise_weight=True, noise_std=noise_std)
                    act_noisy = [h.detach() for h in model_noisy.h]
                pred.append(last_out)

                # Store activities
                activity_hist_0.append(outs[0].cpu().numpy())
                activity_hist_1.append(outs[1].cpu().numpy())

                # Compute and apply decorelation updates online
                if decorrelation:
                    model.update_params(lr_decor=lr_decor)
                    model_noisy.decor_layers[1].weight.data  = copy.deepcopy(model.decor_layers[1].weight.data)

                # Acumulate error difference
                loss = criterion(last_out, y_train[t])
                losses_diff.append(criterion(last_out_noisy, y_train[t]) - criterion(last_out, y_train[t]))
                
                # Compute traces
                et_w0.append(noise[0].transpose(0, 1) @ outs_new[0])
                et_w1.append(noise[1].transpose(0, 1) @ outs_new[1])
                if t > 0:
                    et_r.append(noise[0].transpose(0, 1) @ outs[1])

                # Update hidden states
                outs = copy.deepcopy(outs_new)

                # Track loss
                running_loss += loss.item()

            # Update here
            for t in tqdm(range(lenght_sequence_train)):  
                
                dW0 = lr_fwd * ((1/noise_std) * (torch.sum(torch.stack(losses_diff[t:])))) * et_w0[t]
                model.fwd_layers[0].weight.data = model.fwd_layers[0].weight.data - dW0
                model_noisy.fwd_layers[0].weight.data = copy.deepcopy(model.fwd_layers[0].weight.data)  
                # dW1 = lr_fwd * ((1/noise_std) * (torch.sum(torch.stack(losses_diff[t:])))) * et_w1[t] 
                dW1 = lr_fwd * ((1/noise_std) * 10 * (losses_diff[t])) * et_w1[t]
                model.fwd_layers[1].weight.data = model.fwd_layers[1].weight.data - dW1
                model_noisy.fwd_layers[1].weight.data = copy.deepcopy(model.fwd_layers[1].weight.data)
                if t > 0:
                    dR = lr_fwd * ((1/noise_std) * (torch.sum(torch.stack(losses_diff[t:])))) * et_r[t-1]
                    model.rec_layers[0].weight.data = model.rec_layers[0].weight.data - dR
                    model_noisy.rec_layers[0].weight.data = copy.deepcopy(model.rec_layers[0].weight.data)


            ## --- Testing --- ##

            # Initial stuff
            test_running_loss = 0.0
            pred_test = []

            for t in tqdm(range(lenght_sequence_test)):

                # Input data point
                last_out, outs = model(x_test[t], outs)
                pred_test.append(last_out)

                # Compute loss
                test_loss = criterion(last_out, y_test[t])
                test_running_loss += test_loss.item()

            # Compute loss and acc
            loss = running_loss / len(x_train)
            test_loss = test_running_loss / len(x_test)
            performances["train"]["loss"].append(loss)
            performances["test"]["loss"].append(test_loss)
            print('Train loss', loss)
            print('Test loss:', test_loss)

            # Compute neural activities and cross-correlation
            activity_hist_0 = np.array(activity_hist_0)
            activity_hist_0 = np.reshape(activity_hist_0, (activity_hist_0.shape[0]*activity_hist_0.shape[1], activity_hist_0.shape[2]))
            activity_hist_1 = np.array(activity_hist_1)
            activity_hist_1 = np.reshape(activity_hist_1, (activity_hist_1.shape[0]*activity_hist_1.shape[1], activity_hist_1.shape[2]))
            cov = activity_hist_0.transpose() @ activity_hist_0
            offdiag_cov_0 = (1.0 - np.eye(len(cov)))*cov
            cov = activity_hist_1.transpose() @ activity_hist_1
            offdiag_cov_1 = (1.0 - np.eye(len(cov)))*cov

            # Store correlation layers
            correlation['input'].append(np.mean(offdiag_cov_0))
            correlation['hidden'].append(np.mean(offdiag_cov_1))
            print('Cross-correlation input and hidden:', np.mean(offdiag_cov_0), np.mean(offdiag_cov_1))

            # Squared lower triangular matrix correlation
            sq_corr_hidden = offdiag_cov_1[np.tril_indices(len(offdiag_cov_1), -1)] ** 2
            

            # Track progress in wandb
            wandb.log({"loss train": loss,
                        "loss test": test_loss,
                        "decor input": np.mean(offdiag_cov_0), "decor hidden": np.mean(offdiag_cov_1),
                        "sq tril corr mean": np.mean(sq_corr_hidden), "sq tril corr max": np.max(sq_corr_hidden), "sq tril corr min": np.min(sq_corr_hidden),
                        "sq tril corr median": np.median(sq_corr_hidden), "sq tril corr Q1": np.percentile(sq_corr_hidden, 25), "sq tril corr Q3": np.percentile(sq_corr_hidden, 75)})

    # Close wandb connection
    wandb.finish()

    # Return logs experiments and model
    return performances, correlation, model
