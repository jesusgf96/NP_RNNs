import torch
import torch.nn as nn
import numpy as np



class RNN(nn.Module):
    def __init__(self, net_structure, batch_size, act_func, act_func_out, decor_input=True, seed=42, device=torch.device("cpu")):
        super().__init__()
        self.net_structure = net_structure
        self.decor_layers = []
        self.rec_layers = []
        self.fwd_layers = []
        self.x = []
        self.x_hat = []
        self.dD_hist = []
        self.h = []
        self.eyes = []
        self.batch_size = batch_size
        self.act_func = act_func
        self.act_func_out = act_func_out
        self.decor_input = decor_input
        self.device = device
        torch.manual_seed(seed)

        # Create layers and states
        self.h.append(None)
        self.x.append(None)
        self.x_hat.append(None)
        self.eyes.append(torch.eye(self.net_structure[0]))
        for indx in range(len(self.net_structure) - 1):
            self.h.append(None)
            self.x.append(None)
            self.x_hat.append(None)
            self.dD_hist.append([])
            self.eyes.append(torch.eye(self.net_structure[indx+1]))
            self.decor_layers.append(nn.Linear(self.net_structure[indx], net_structure[indx], bias=False))
            self.rec_layers.append(nn.Linear(self.net_structure[indx+1], net_structure[indx+1], bias=False))
            self.fwd_layers.append(nn.Linear(net_structure[indx], net_structure[indx + 1], bias=False))
            torch.nn.init.eye_(self.decor_layers[-1].weight)
            # torch.nn.init.eye_(self.rec_layers[-1].weight) # Identity init for recurrent connection
            torch.nn.init.xavier_normal_(self.rec_layers[-1].weight) # Xavier init for recurrent connection
            torch.nn.init.xavier_normal_(self.fwd_layers[-1].weight)

        # Exclude decorrelation layers from backpropagation the update
        for indx in range(len(self.net_structure) - 1):
            for param in self.decor_layers[indx].parameters():
                param.requires_grad = False

        # This allows pytorch to manage moving everything to device
        self.decors_mod = nn.ModuleList(self.decor_layers)
        self.rec_mod = nn.ModuleList(self.rec_layers)
        self.fwd_mod = nn.ModuleList(self.fwd_layers)

        # Reset states
        self.reset_states()


    # Reset states and adjust the batch size
    def reset_states(self):
        self.dD_hist = []
        for indx in range(len(self.net_structure)):
            self.dD_hist.append([])
            self.x[indx] = torch.zeros(
                self.batch_size, self.net_structure[indx]
            ).to(self.device)
            self.x_hat[indx] = torch.zeros(
                self.batch_size, self.net_structure[indx]
            ).to(self.device)
            self.h[indx] = torch.zeros(
                self.batch_size, self.net_structure[indx]
            ).to(self.device)
            self.eyes[indx] = self.eyes[indx].to(self.device)
        return [torch.zeros(self.batch_size, self.net_structure[indx]).to(self.device) for indx in range(len(self.net_structure))]



    # Foward pass
    def forward(self, input_data, outs, noise_node=False, noise_weight=False, noise_std=10e-6):

        # Noise per layer
        noises = []

        # Decorrelate linear input
        self.h[0] = self.x[0] = input_data
        self.x_hat[0] = self.x[0] @ self.decor_layers[0].weight.transpose(0, 1)

        # Iterating recurrent layers
        for indx in range(len(self.net_structure) - 2):

              # Forward + recurrent transformation
              if noise_weight:
                  noises.append(torch.randn_like(self.fwd_layers[indx].weight, device=self.device) * noise_std) # fwd
                  noises.append(torch.randn_like(self.rec_layers[indx].weight, device=self.device) * noise_std) # recurrent
                  self.h[indx+1] = self.x_hat[indx] @ (self.fwd_layers[indx].weight + noises[-2]).transpose(0, 1) + outs[indx+1] @ (self.rec_layers[indx].weight + noises[-1]).transpose(0, 1)
              else:
                  self.h[indx+1] = self.x_hat[indx] @ self.fwd_layers[indx].weight.transpose(0, 1) + outs[indx+1] @ self.rec_layers[indx].weight.transpose(0, 1)

              # Adding noise into the activation
              if noise_node:
                  noises.append(torch.randn_like(self.h[indx+1], device=self.device) * noise_std)
                  self.h[indx+1] = self.h[indx+1] + noises[-1]
              self.x[indx+1] = self.act_func(self.h[indx+1])
              self.x_hat[indx+1] = self.x[indx+1] @ self.decor_layers[indx+1].weight.transpose(0, 1)

        # Final layer
        if noise_weight:
            noises.append(torch.randn_like(self.fwd_layers[-1].weight, device=self.device) * noise_std)
            self.h[-1] = self.x_hat[-2] @ (self.fwd_layers[-1].weight + noises[-1]).transpose(0, 1)
        else:
            self.h[-1] = self.x_hat[-2] @ self.fwd_layers[-1].weight.transpose(0, 1)
        if noise_node:
            noises.append(torch.randn_like(self.h[-1], device=self.device) * noise_std)
            self.h[-1] = self.h[-1] + noises[-1]
        self.x[-1] =  self.x_hat[-1] = self.act_func_out(self.h[-1])

        # Return output network and each layer
        if noise_node or noise_weight:
            return self.x[-1], [x_hat.detach() for x_hat in self.x_hat], noises
        else:
            return self.x[-1], [x_hat.detach() for x_hat in self.x_hat],



    # Node perturbation updating rule
    def node_perturbation(self, input_sequence, targets, loss, lr, noise_std=10e-6):

        # States of the network over time
        lenght_sequence = input_sequence.shape[0]
        u = torch.zeros(lenght_sequence, self.batch_size, self.net_structure[0], device=self.device)
        x = torch.zeros(lenght_sequence, self.batch_size, self.net_structure[1], device=self.device)
        y = torch.zeros(lenght_sequence, self.batch_size, self.net_structure[2], device=self.device)
        losses = torch.zeros(lenght_sequence, device=self.device)
        u_noisy = torch.zeros(lenght_sequence, self.batch_size, self.net_structure[0], device=self.device)
        x_noisy = torch.zeros(lenght_sequence, self.batch_size, self.net_structure[1], device=self.device)
        y_noisy = torch.zeros(lenght_sequence, self.batch_size, self.net_structure[2], device=self.device)
        losses_noisy = torch.zeros(lenght_sequence, device=self.device)
        noise_x = torch.zeros(lenght_sequence, self.batch_size, self.net_structure[1], device=self.device)
        noise_y = torch.zeros(lenght_sequence, self.batch_size, self.net_structure[2], device=self.device)
        dims_target = len(targets.size())

        # Forward pass over the sequence without noise
        outs = self.reset_states()
        for t in range(lenght_sequence):
            last_out, outs = self(input_sequence[t], outs)
            u[t] = self.x_hat[0]
            x[t] = self.x_hat[1]
            y[t] = self.x_hat[2]
            if dims_target < 3:
                losses[t] = loss(targets, last_out) # One single label per sequence
            else:
                losses[t] = loss(targets[t], last_out) # One label per timestep

        # Forward pass over the sequence with noise (noise not applied to the input)
        outs = self.reset_states()
        for t in range(lenght_sequence):
            last_out, outs, noises = self(input_sequence[t], outs, noise_node=True, noise_std=noise_std)
            u_noisy[t] = self.x_hat[0]
            x_noisy[t] = self.x_hat[1]
            y_noisy[t] = self.x_hat[2]
            if dims_target < 3:
                losses_noisy[t] = loss(targets, last_out) # One single label per sequence
            else:
                losses_noisy[t] = loss(targets[t], last_out) # One label per timestep
            noise_x[t] = noises[0]
            noise_y[t] = noises[1]

        # Update weights over time
        for t in range(lenght_sequence):
            dW0 = lr * ((1/noise_std) * (losses_noisy[t] - losses[t])) * noise_x[t].transpose(0, 1) @ u[t]
            self.fwd_layers[0].weight.data = self.fwd_layers[0].weight.data - dW0
            dW1 = lr * ((1/noise_std) * (losses_noisy[t] - losses[t])) * noise_y[t].transpose(0, 1) @ x[t]
            self.fwd_layers[1].weight.data = self.fwd_layers[1].weight.data - dW1
            if t > 0:
                dR = lr * ((1/noise_std) * (losses_noisy[t] - losses[t])) * noise_x[t].transpose(0, 1) @ x[t-1]
                self.rec_layers[0].weight.data = self.rec_layers[0].weight.data - dR



    # Node perturbation updating rule
    def weight_perturbation(self, input_sequence, targets, loss, lr, noise_std=10e-6):

        # States of the network over time
        lenght_sequence = input_sequence.shape[0]
        losses = torch.zeros(lenght_sequence, device=self.device)
        losses_noisy = torch.zeros(lenght_sequence, device=self.device)
        noise_fwd1 = torch.zeros(lenght_sequence, self.net_structure[1], self.net_structure[0], device=self.device)
        noise_fwd2 = torch.zeros(lenght_sequence, self.net_structure[2], self.net_structure[1], device=self.device)
        noise_rec = torch.zeros(lenght_sequence, self.net_structure[1], self.net_structure[1], device=self.device)
        dims_target = len(targets.size())

        # Forward pass over the sequence without noise
        outs = self.reset_states()
        for t in range(lenght_sequence):
            last_out, outs = self(input_sequence[t], outs)
            if dims_target < 3:
                losses[t] = loss(targets, last_out) # One single label per sequence
            else:
                losses[t] = loss(targets[t], last_out) # One label per timestep

        # Forward pass over the sequence with noise (noise not applied to the input)
        outs = self.reset_states()
        for t in range(lenght_sequence):
            last_out, outs, noises = self(input_sequence[t], outs, noise_weight=True, noise_std=noise_std)
            if dims_target < 3:
                losses_noisy[t] = loss(targets, last_out) # One single label per sequence
            else:
                losses_noisy[t] = loss(targets[t], last_out) # One label per timestep
            noise_fwd1[t] = noises[0]
            noise_fwd2[t] = noises[2]
            noise_rec[t] = noises[1]

        # Update weights over time
        for t in range(lenght_sequence):
            dW0 = lr * ((1/noise_std) * (losses_noisy[t] - losses[t])) * noise_fwd1[t]
            self.fwd_layers[0].weight.data = self.fwd_layers[0].weight.data - dW0
            dW1 = lr * ((1/noise_std) * (losses_noisy[t] - losses[t])) * noise_fwd2[t]
            self.fwd_layers[1].weight.data = self.fwd_layers[1].weight.data - dW1
            if t > 0:
                dR = lr * ((1/noise_std) * (losses_noisy[t] - losses[t])) * noise_rec[t]
                self.rec_layers[0].weight.data = self.rec_layers[0].weight.data - dR


    # Node perturbation updating rule
    def node_perturbation_activity(self, input_sequence, targets, loss, lr, noise_std=10e-6):

        # States of the network over time
        lenght_sequence = input_sequence.shape[0]
        u = torch.zeros(lenght_sequence, self.batch_size, self.net_structure[0], device=self.device)
        x = torch.zeros(lenght_sequence, self.batch_size, self.net_structure[1], device=self.device)
        h_x = torch.zeros(lenght_sequence, self.batch_size, self.net_structure[1], device=self.device)
        h_y = torch.zeros(lenght_sequence, self.batch_size, self.net_structure[2], device=self.device)
        losses = torch.zeros(lenght_sequence, device=self.device)
        h_x_noisy = torch.zeros(lenght_sequence, self.batch_size, self.net_structure[1], device=self.device)
        h_y_noisy = torch.zeros(lenght_sequence, self.batch_size, self.net_structure[2], device=self.device)
        losses_noisy = torch.zeros(lenght_sequence, device=self.device)
        dims_target = len(targets.size())

        # Forward pass over the sequence without noise
        outs = self.reset_states()
        for t in range(lenght_sequence):
            last_out, outs = self(input_sequence[t], outs)
            u[t] = self.x_hat[0]
            x[t] = self.x_hat[1]
            h_x[t] = self.h[1]
            h_y[t] = self.h[2]
            if dims_target < 3:
                losses[t] = loss(targets, last_out) # One single label per sequence
            else:
                losses[t] = loss(targets[t], last_out) # One label per timestep

        # Forward pass over the sequence with noise (noise not applied to the input)
        outs_noisy = self.reset_states()
        for t in range(lenght_sequence):
            last_out_noisy, outs_noisy, noises = self(input_sequence[t], outs_noisy, noise_node=True, noise_std=noise_std)
            h_x_noisy[t] = self.h[1]
            h_y_noisy[t] = self.h[2]
            if dims_target < 3:
                losses_noisy[t] = loss(targets, last_out_noisy) # One single label per sequence
            else:
                losses_noisy[t] = loss(targets[t], last_out_noisy) # One label per timestep

        # Update weights over time
        N = torch.tensor(np.sum(self.net_structure), device=self.device)
        for t in range(lenght_sequence):

            dW0 = (torch.sqrt(N) * (losses_noisy[t] - losses[t]) * ((h_x_noisy[t] - h_x[t]) / (torch.norm(h_x_noisy[t] - h_x[t], 'fro')**2))).transpose(0, 1) @ u[t]
            self.fwd_layers[0].weight.data = self.fwd_layers[0].weight.data - lr * dW0
            dW1 = (torch.sqrt(N) * (losses_noisy[t] - losses[t]) * ((h_y_noisy[t] - h_y[t]) / (torch.norm(h_y_noisy[t] - h_y[t], 'fro')**2))).transpose(0, 1) @ x[t]
            self.fwd_layers[1].weight.data = self.fwd_layers[1].weight.data - lr * dW1
            if t > 0:
                dR = (torch.sqrt(N) * (losses_noisy[t] - losses[t]) * ((h_x_noisy[t] - h_x[t]) / (torch.norm(h_x_noisy[t] - h_x[t], 'fro')**2))).transpose(0, 1) @ x[t-1]
                self.rec_layers[0].weight.data = self.rec_layers[0].weight.data - lr * dR

        return last_out, outs


    # Compute decorrelation parameters updates
    def update_params(self, lr_decor):

        # Decorrelate hidden + input?
        if self.decor_input:
            it_range = range(len(self.net_structure) - 1)
        else:
            it_range = range(1, len(self.net_structure) - 1)

        # Iterating layers
        for indx in it_range:

            # Decorrelation weights
            corr = (1/self.batch_size)*torch.einsum('ni,nj->ij', self.x_hat[indx], self.x_hat[indx])*(1.0 - self.eyes[indx])
            dD = (torch.einsum('ij,jk->ik', corr, self.decor_layers[indx].weight))
            dD = lr_decor * corr

            # Apply updates online
            self.decor_layers[indx].weight.data = self.decor_layers[indx].weight.data - dD


    # Compute decorrelation parameters updates
    def compute_update_params(self, lr_decor):

        # Decorrelate hidden + input?
        if self.decor_input:
            it_range = range(len(self.net_structure) - 1)
        else:
            it_range = range(1, len(self.net_structure) - 1)

        # Iterating layers
        for indx in it_range:

            # Decorrelation weights
            corr = (1/self.batch_size)*torch.einsum('ni,nj->ij', self.x_hat[indx], self.x_hat[indx])*(1.0 - self.eyes[indx])
            dD = (torch.einsum('ij,jk->ik', corr, self.decor_layers[indx].weight))
            dD = lr_decor * corr

            # Store updates but don't apply them
            self.dD_hist[indx].append(dD)


    # Apply decorrelation parameters updates
    def apply_update_params(self):

        # Decorrelate hidden + input?
        if self.decor_input:
            it_range = range(len(self.net_structure) - 1)
        else:
            it_range = range(1, len(self.net_structure) - 1)

        # Iterate layers
        for indx in it_range:

            # Average update over time
            dD = torch.mean(torch.stack(self.dD_hist[indx]), axis=0)
            self.decor_layers[indx].weight.data = self.decor_layers[indx].weight.data - dD
