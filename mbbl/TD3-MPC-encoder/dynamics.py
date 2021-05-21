#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
from torch_util import device, FLOAT

import numpy as np

def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

class MLPnetwork1(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden=128, activation=nn.LeakyReLU):
        super(MLPnetwork1, self).__init__()

        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden

        self.value = nn.Sequential(
            nn.Linear(dim_input, self.dim_hidden),
            activation(),
            nn.Linear(self.dim_hidden, self.dim_hidden),
            activation(),
            nn.Linear(self.dim_hidden, self.dim_output)
        )

        self.value.apply(init_weight)

    def forward(self, inputs):
        value = self.value(inputs)
        return value.squeeze()


class GaussianModel1(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden=128, max_sigma=1e1, min_sigma=1e-4):

        super().__init__()

        self.fc = nn.Linear(dim_input, dim_hidden)
        self.ln = nn.LayerNorm(dim_hidden)
        self.fc_mu = nn.Linear(dim_hidden, dim_output)
        self.fc_sigma = nn.Linear(dim_hidden, dim_output)

        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)
        nn.init.xavier_normal_(self.fc_mu.weight)
        nn.init.constant_(self.fc_mu.bias, 0.0)
        nn.init.xavier_normal_(self.fc_sigma.weight)
        nn.init.constant_(self.fc_sigma.bias, 0.0)

        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        assert self.max_sigma >= self.min_sigma

    def forward(self, input1):
        x = input1
        x = self.fc(x)
        x = self.ln(x)
        x = torch.relu(x)
        mu = self.fc_mu(x)
        sigma = torch.sigmoid(self.fc_sigma(x))
        sigma = (
            self.min_sigma + (self.max_sigma - self.min_sigma) * sigma
        )
        return mu, sigma

    def sample_prediction(self, input1):
        mu, sigma = self(input1)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

class GaussianModel2(nn.Module):
    def __init__(self, dim_input1, dim_input2, dim_output, dim_hidden=128, max_sigma=1e1, min_sigma=1e-4):

        super().__init__()

        self.fc = nn.Linear(dim_input1 + dim_input2, dim_hidden)
        self.ln = nn.LayerNorm(dim_hidden)
        self.fc_mu = nn.Linear(dim_hidden, dim_output)
        self.fc_sigma = nn.Linear(dim_hidden, dim_output)

        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)
        nn.init.xavier_normal_(self.fc_mu.weight)
        nn.init.constant_(self.fc_mu.bias, 0.0)
        nn.init.xavier_normal_(self.fc_sigma.weight)
        nn.init.constant_(self.fc_sigma.bias, 0.0)

        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        assert self.max_sigma >= self.min_sigma

    def forward(self, input1, input2):
        x = torch.cat([input1, input2], dim=-1)
        x = self.fc(x)
        x = self.ln(x)
        x = torch.relu(x)
        mu = self.fc_mu(x)
        sigma = torch.sigmoid(self.fc_sigma(x))
        sigma = (
            self.min_sigma + (self.max_sigma - self.min_sigma) * sigma
        )
        return mu, sigma

    def sample_prediction(self, input1, input2):
        mu, sigma = self(input1, input2)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

class Encoder():

    def __init__(self, state_dim, latent_dim, action_dim):

        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.encoder = MLPnetwork1(self.state_dim, self.latent_dim, dim_hidden=128).to(device)
        self.decoder = MLPnetwork1(self.latent_dim, self.state_dim, dim_hidden=128).to(device)
        self.forward_dynamics = GaussianModel2(self.latent_dim, self.action_dim, self.latent_dim, dim_hidden=128).to(device)
        self.inverse_dynamics = GaussianModel2(self.latent_dim, self.latent_dim, self.action_dim, dim_hidden=128).to(device)

        self.optimizer = optim.Adam(list(self.encoder.parameters())+list(self.decoder.parameters()), lr=0.0003)
        self.optimizer_fdyn = optim.Adam(self.forward_dynamics.parameters(), lr=0.0003)
        self.optimizer_idyn = optim.Adam(self.inverse_dynamics.parameters(), lr=0.0003)

        self.loss = 0
        self.fdyn_loss = 0
        self.idyn_loss = 0

    def update_encoder(self, state1, action1, reward1, next_state1, state2, action2, reward2, next_state2):

        encoded_state1 = self.encoder.forward(state1)
        encoded_nstate1 = self.encoder.forward(next_state1)
        pred_encoded_nstate1 = self.forward_dynamics.sample_prediction(encoded_state1, action1)
        pred_encoded_action1 = self.inverse_dynamics.sample_prediction(encoded_state1, pred_encoded_nstate1)
        decoded_state1 = self.decoder.forward(pred_encoded_nstate1)
        decoded_nstate1 = self.decoder.forward(pred_encoded_nstate1)
        
        encoded_state2 = self.encoder.forward(state2)
        encoded_nstate2 = self.encoder.forward(next_state2)
        pred_encoded_nstate2 = self.forward_dynamics.sample_prediction(encoded_state2, action2)
        pred_encoded_action2 = self.inverse_dynamics.sample_prediction(encoded_state2, pred_encoded_nstate2)
        decoded_state2 = self.decoder.forward(pred_encoded_nstate2)
        decoded_nstate2 = self.decoder.forward(pred_encoded_nstate2)
                
        self.fdyn_loss = nn.MSELoss()(pred_encoded_nstate1, encoded_nstate1) + nn.MSELoss()(pred_encoded_nstate2, encoded_nstate2)
        self.idyn_loss = nn.MSELoss()(pred_encoded_action1, action1) + nn.MSELoss()(pred_encoded_action2, action2)   
        self.loss = nn.MSELoss()(state1,decoded_state1)+nn.MSELoss()(next_state1,decoded_nstate1)+self.fdyn_loss+self.idyn_loss
        
        self.optimizer_fdyn.zero_grad()
        self.optimizer_idyn.zero_grad()
        self.optimizer.zero_grad()

        self.fdyn_loss.backward(retain_graph=True)
        self.idyn_loss.backward(retain_graph=True)
        self.loss.backward(retain_graph=True)   

        self.optimizer_fdyn.step()
        self.optimizer_idyn.step()
        self.optimizer.step()
        
    def intrisic_reward(self, state, action, next_state):

        with torch.no_grad():
            encoded_state = self.encoder.forward(state)
            pred_encoded_nstate = self.forward_dynamics.sample_prediction(encoded_state, action)
            encoded_nstate = self.encoder.forward(next_state)

        return torch.linalg.norm((pred_encoded_nstate - encoded_state), ord=1)




    def update_writer(self, writer, i_iter):

        enco_loss = torch.mean(self.loss)
        deco_loss = torch.mean(self.loss)
        fd_loss = torch.mean(self.fdyn_loss)
        id_loss = torch.mean(self.idyn_loss)

        writer.add_scalar("encodings/encoder_loss", enco_loss, i_iter)
        writer.add_scalar("encodings/decoder_loss", enco_loss, i_iter)
        writer.add_scalar("encodings/forward_dynamics_loss", fd_loss, i_iter)
        writer.add_scalar("encodings/inverse_dynamics_loss", id_loss, i_iter)

        return writer

   
    def predict_state_trajectory(self, curr_states, controls):
        next_states=[]
        
        for i in range(len(controls)):
             
            curr_state = curr_states[i]
            #curr_state = FLOAT(curr_state[i].unsqueeze(0).cuda())
            #controls[i] = FLOAT(controls[i].unsqueeze(0).to(device))    
            with torch.no_grad():

                #self.encoder.is_cpu()
                curr_state_features = self.encoder.forward(curr_state)
                pred_next_state_features = self.forward_dynamics.sample_prediction(curr_state_features,controls[i])
                predicted_state = self.decoder.forward(pred_next_state_features)
                #next_states.append(predicted_state.cpu().numpy()[0])
                next_states.append(predicted_state)

        return next_states