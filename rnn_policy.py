import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np

def get_init_rnn_state(num_rnn_layers, hidden_dim, batch_size, is_training, device):
    if is_training:
        hidden = torch.zeros(num_rnn_layers, batch_size, hidden_dim).to(device)
    else:
        hidden = torch.zeros(num_rnn_layers, 1, hidden_dim).to(device)
    return hidden        

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class RNNPolicy(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        # information        
        self.is_continuous = args.is_continuous
        self.state_dim = args.state_dim
        self.linear_dim = args.linear_dim
        self.hidden_dim = args.hidden_dim
        self.action_dim = args.action_dim
        self.batch_size = args.batch_size
        self.device = torch.device(args.device)
        self.num_rnn_layers = args.num_rnn_layers
        
        # network
        self.fc = layer_init(nn.Linear(args.state_dim, args.linear_dim))
        self.gru = nn.GRU(args.linear_dim, args.hidden_dim, \
                            num_layers=self.num_rnn_layers, bias=True)
        if self.is_continuous:
            self.mean = layer_init(nn.Linear(self.hidden_dim, self.action_dim))
            self.std =  layer_init(nn.Linear(self.hidden_dim, self.action_dim))
        else:
            self.policy_logits = layer_init(nn.Linear(args.hidden_dim, self.action_dim))
        
    def _format(self, state, device):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            x = x.to(device=device)
        else:
            x = x.to(device=device)
        return x
    
    def forward(self, state, hidden=None):
        state = self._format(state, self.device)
        if len(state.shape) < 3:
            state = state.reshape(1, 1, -1)
        x = F.leaky_relu(self.fc(state))
        if len(hidden.shape) == 4:
            hidden = hidden.permute(0, 2, 1, 3)
            hidden = hidden[0].contiguous()
        x, new_hidden = self.gru(x, hidden)
        x, new_hidden = torch.tanh(x), torch.tanh(hidden)
        if self.is_continuous:
            mu = torch.tanh(self.mean(x))
            std = F.softplus(self.std(x))
            dist = Normal(mu, std)
        else:
            logits = self.policy_logits(x)
            prob = F.softmax(logits, dim=-1)
            dist = Categorical(prob)
        return dist, new_hidden

    def choose_action(self, state, hidden, is_training=False):
        dist, hidden = self.forward(state, hidden)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob = log_prob.sum(-1)
        if is_training:
            return action, dist.log_prob(action), hidden
        else:
            while len(action.shape) != 1:
                action = action.squeeze(0)
                log_prob = log_prob.squeeze(0)
            if self.is_continuous:
                return action.detach().to('cpu').numpy(), log_prob.detach().to('cpu').numpy(), hidden
            else:
                return action.detach().to('cpu').numpy().item(), log_prob.detach().to('cpu').numpy(), hidden
    
class RNNCritic(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        # information        
        self.is_continuous = args.is_continuous
        self.state_dim = args.state_dim
        self.linear_dim = args.linear_dim
        self.hidden_dim = args.hidden_dim
        self.action_dim = args.action_dim
        self.batch_size = args.batch_size
        self.device = torch.device(args.device)
        self.num_rnn_layers = args.num_rnn_layers
        
        # network
        self.fc = nn.Linear(args.state_dim, args.linear_dim)
        self.gru = nn.GRU(args.linear_dim, args.hidden_dim, \
                            num_layers=self.num_rnn_layers, bias=True)
        self.v = nn.Linear(args.hidden_dim, 1)   
             
    def _format(self, state, device):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            x = x.to(device=device)
        else:
            x = x.to(device=device)
        return x
    
    def forward(self, state, hidden=None):
        state = self._format(state, self.device)
        if len(state.shape) < 3:
            state = state.reshape(1, 1, -1)
        if len(hidden.shape) == 4:
            hidden = hidden.permute(0, 2, 1, 3)
        x = F.leaky_relu(self.fc(state))
        x, new_hidden = self.gru(x, hidden)
        x, new_hidden = torch.tanh(x), torch.tanh(hidden)
        return self.v(x), new_hidden
    
    def value(self, state, hidden):
        value = self.forward(state, hidden)
        return value, hidden