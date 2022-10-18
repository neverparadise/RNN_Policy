import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class TupleEmbedding(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        input_dim = args.state_dim + args.action_dim + 2 # 2: reward, done dimension
        self.embedding = nn.Sequential(
            nn.Linear(args.input_dim, args.hidden_dim),
            nn.LeaykyReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.LeaykyReLU(),
            nn.Linear(args.hidden_dim, args.output_dim),
        )
        
    def forward(self, transition):
        state, action, reward, done = transition
        concatenated = torch.cat([state, action, reward, done], dim=-1)
        return self.embedding(concatenated)

class PPOAlgorithm:
    def __init__(self) -> None:
        pass

class GaussianPolicy(nn.Module):
    def __init__(self, args):
        super(GaussianPolicy, self).__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.mean_layer = nn.Linear(args.hidden_dim, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))  # We use 'nn.Parameter' to train log_std automatically
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            torch.nn.orthogonal_init(self.fc1)
            torch.nn.orthogonal_init(self.fc2)
            torch.nn.orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mean = self.max_action * torch.tanh(self.mean_layer(s))  # [-1,1]->[-max_action,max_action]
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist

class ValueNetwork(nn.Module):
    def __init__(self, args):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            torch.nn.orthogonal_init(self.fc1)
            torch.nn.orthogonal_init(self.fc2)
            torch.nn.orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s
    
    
def get_init_lstm_state(num_rnn_layers, lstm_dim, batch_size, is_training, device):
    if is_training:
        hc_state = (torch.zeros(num_rnn_layers, batch_size, lstm_dim).to(device),
                    torch.zeros(num_rnn_layers, batch_size, lstm_dim).to(device))
    else:
        hc_state = (torch.zeros(num_rnn_layers, 1, lstm_dim).to(device),
                    torch.zeros(num_rnn_layers, 1, lstm_dim).to(device))
    return hc_state        


class RNNGaussianPolicy(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.is_continuous = args.is_continuous
        self.action_dim = args.action_dim
        self.linear_dim = args.linear_dim
        self.batch_size = args.batch_size
        self.device = args.device
        self.num_recurrent_layers = args.num_recurrent_layers
        self.lstm = nn.LSTM(args.state_dim, args.lstm_dim, num_layers=self.num_recurrent_layers, bias=True)
        self.fc = nn.Linear(args.lstm_dim, args.linear_dim)
        self.policy_logits = nn.Linear(args.linear_dim, self.action_dim)
        self.mean = nn.Linear(self.linear_dim, self.action_dim)
        self.logstd =  nn.Linear(self.linear_dim, self.action_dim)

    def forward(self, state, hidden_state):
        if len(state.shape) > 3: # ! (L, B, F)
            hs, cs = get_init_lstm_state(self.num_recurrent_layers, self.lstm_dim, self.batch_size, 
                                                  is_training=True, device=self.device)
            x_sequence = []
            for x_t in state:
                new_hs, new_cs = self.lstm(x_t, (hs, cs))
                
                
                

        
        
class RNNValueNetwork(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.action_dim = args.action_dim
        self.linear_dim = args.linear_dim
        self.num_recurrent_layers = args.num_recurrent_layers
        self.lstm = nn.LSTM(args.state_dim, args.lstm_dim, num_layers=args.num_recurrent_layers)
        self.fc = nn.Linear(args.lstm_dim, args.linear_dim)
        self.value = nn.Linear(self.linear_dim)
    
    def forward(self, state):
        if len(state.shape) > 3:
            init_lstm_state = get_init_lstm_state(self.num_recurrent_layers, self.lstm_dim, self.batch_size, 
                                                  is_training=True, device=self.device)
            
    
class RL2PolicyNetwork(nn.Module):

    """
    LSTM
    input: (Batch, Sequence, Feature) or (Sequence, Batch, Feature)
    output: (Batch, Action)
   
    
    """
    def __init__(self, args) -> None:
        super().__init__()
        self.phi = TupleEmbedding(args)
        

class PPO_continuous():
    def __init__(self, args):
        self.max_action = args.max_action
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.actor = GaussianPolicy(args)
        self.critic = ValueNetwork(args)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def evaluate(self, s):  # When evaluating the policy, we only use the mean
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a = self.actor(s).detach().numpy().flatten()
        return a

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        with torch.no_grad():
            dist = self.actor.get_dist(s)
            a = dist.sample()  # Sample the action according to the probability distribution
            a = torch.clamp(a, -self.max_action, self.max_action)  # [-max,max]
            a_logprob = dist.log_prob(a)  # The log probability density of the action
        return a.numpy().flatten(), a_logprob.numpy().flatten()

    def update(self, replay_buffer, total_steps):
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()  # Get training data
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = self.actor.get_dist(s[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index])
                # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True))  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now