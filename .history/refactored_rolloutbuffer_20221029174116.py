import gym
import torch
import numpy as np
from dataclasses import asdict, dataclass
import torch as th
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Generator, Optional, Tuple, Callable, Dict, List, NamedTuple, Union

def to_tensor(np_array, device="cpu"):
    if isinstance(np_array, np.ndarray):
        return torch.from_numpy(np_array).float().to(device)
    return np_array.float().to(device)


def to_numpy(data):
    if isinstance(data, tuple):
        return tuple(to_numpy(x) for x in data)
    if isinstance(data, torch.autograd.Variable):
        return data.to('cpu').detach().numpy()
    return data

# RecurrentRolloutBufferSamples(*tuple([to_tensor(sample, device=self.device) for sample in samples]))

@dataclass
class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    values: th.Tensor
    returns: th.Tensor
    log_probs: th.Tensor
    advantages: th.Tensor
    
    
    
@dataclass
class RecurrentRolloutBufferSamples:
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    returns: th.Tensor
    old_log_probs: th.Tensor
    advantages: th.Tensor
    masks:  th.Tensor
    actor_hxs: th.Tensor
    critic_hxs: th.Tensor
    
    def get_transition(self):
        return self.observations, self.actions, self.old_values, self.returns, \
            self.old_log_probs, self.advantages, self.masks, self.actor_hxs, self.critic_hxs
            
            
class RolloutBuffer:
    def __init__(
        self,
        buffer_size: int,
        state_dim: Tuple[int],
        action_space: gym.Space,
        gamma: float,
        gae_lambda: float,
        device: str,
        is_recurrent: bool,
        recurrent_size: int = None,
        num_rnn_layers: int = 1,
        is_shared_network=False
        ):
        
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_space = action_space
        self.action_dim = (action_space.n,) if isinstance(action_space, gym.spaces.Discrete) else action_space.shape
        self.is_continuous = False if isinstance(self.action_space, gym.spaces.Discrete) else True
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.is_recurrent = is_recurrent
        self.is_shared_network = is_shared_network
        
        if self.is_recurrent:
            self.recurrent_size = recurrent_size
            self.num_rnn_layers = num_rnn_layers
        
        self.reset()
    
    def reset(self):
        """
            Resets the buffer.
        """
        self.observations = np.zeros((self.buffer_size, *self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, *self.action_dim), dtype=np.float32)
        self.values = np.zeros((self.buffer_size,), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size,), dtype=np.float32)
        self.terminals = np.zeros((self.buffer_size,), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size,), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size,), dtype=np.float32)

        if self.is_recurrent:
            if self.is_shared_network:
                self.hiddens = np.zeros((self.buffer_size, self.num_rnn_layers, self.recurrent_size), dtype=np.float32)
            else:
                # 하나의 시퀀스에 대한 hidden state를 저장한다?
                self.actor_hxs = np.zeros((self.buffer_size, self.num_rnn_layers, self.recurrent_size), dtype=np.float32)
                self.critic_hxs = np.zeros((self.buffer_size, self.num_rnn_layers, self.recurrent_size), dtype=np.float32)
            
        # List for tracking the trajectory indices because we don't know
        # how many we can collect
        self.trajectory_index = np.array([0])
        self.pt = 0
        
    def add_sample(self, sample: Tuple[Any]) -> bool:
        """
            Adds a sample to the buffer and increments pointer.
            :param sample: Environment interaction for a single timestep;
                stores state value and log prob of action.
            :return: Whether buffer full or not
        """

        if self.pt != self.buffer_size:
            sample = tuple([to_numpy(item) for item in sample])
            if not self.is_recurrent:
                observation, action, value, reward, log_prob, terminal = sample
            else:
                observation, action, value, reward, log_prob, terminal, actor_hx, critic_hx = sample
            
            # 1. Observation
            self.observations[self.pt] = observation
            
            # 2. Action
            if self.is_continuous: # Continuous
                self.actions[self.pt] = action
            else: # Discrete
                self.actions[self.pt][action] = 1 # if discrete action, action is index
            
            # Value, Reward, Log prob, Done
            self.values[self.pt] = value
            self.rewards[self.pt] = reward
            self.log_probs[self.pt] = log_prob
            self.terminals[self.pt] = terminal

            if self.is_recurrent:
                # hidden_shape = (num_rnn_layers, batch_size, hidden_size)
                # we need to squeeze batch dim
                if self.is_shared_network:
                    self.hiddens[self.pt] = actor_hx.squeeze(1)
                else:
                    self.actor_hxs[self.pt] = actor_hx.squeeze(1)
                    self.critic_hxs[self.pt] = critic_hx.squeeze(1)
            self.pt += 1
        
        return self.pt == self.buffer_size
        
    def end_trajectory(self, last_value: np.ndarray, last_done: int):
        """
            Ends trajectory calculating GAE https://arxiv.org/abs/1506.02438 and 
            the lambda-return (TD(lambda) estimate).
            The TD(lambda) estimator has also two special cases:
            - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
            - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))
            :param last_value: Last state value for observation resulting from final steps
            :param last_done: Whether final state was terminal or not.
        """
        # First create trajectory_advantage array to hold
        # trajectory length entries
        self.trajectory_index = np.concatenate((self.trajectory_index, np.array([self.pt]))) # add last pointer
        traj_range = np.arange(self.trajectory_index[-2], self.trajectory_index[-1])
        last_advantage = 0
        for step in reversed(traj_range):
            if step == traj_range[-1]:
                next_value = last_value
                next_non_terminal = 1.0 - last_done
            else: # step == traj_range[-2]
                next_value = self.values[step + 1]
                next_non_terminal = self.terminals[step + 1]
            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            last_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            self.advantages[step] = last_advantage
        # https://github.com/DLR-RM/stable-baselines3/blob/6f822b9ed7d6e8f57e5a58059923a5b24e8db283/stable_baselines3/common/buffers.py#L347-L348
        self.returns[traj_range] = self.advantages[traj_range] + self.values[traj_range]
    
                    
    def get_samples(self, indices: np.ndarray) -> RolloutBufferSamples:
        """
            Returns samples as RolloutBufferSamples for training.
            
            :param indices: Indices for indexing data.
        """
        samples = (
            self.observations[indices],
            self.actions[indices],
            self.values[indices],
            self.returns[indices],
            self.log_probs[indices],
            self.advantages[indices]
        )

        return RolloutBufferSamples(*tuple([to_tensor(sample, device=self.device) for sample in samples]))

    def get_samples_recurrent(self, indices: np.ndarray) -> RecurrentRolloutBufferSamples:
        """
            Returns samples as RolloutBufferSamples for training.
            
            :param indices: Indices for indexing data.
        """
        samples = [
            self.observations[indices],
            self.actions[indices],
            self.values[indices],
            self.returns[indices],
            self.log_probs[indices],
            self.advantages[indices],
            np.ones(len(indices)),
         ]
        if self.is_shared_network:
            samples.extend([self.hiddens[indices]])
        else:
            samples.extend([self.actor_hxs[indices], self.critic_hxs[indices]])
        sample_list = [to_tensor(sample, device=self.device) for sample in samples]
        return RecurrentRolloutBufferSamples(*tuple(sample_list))
    
    def sample_transtions(self, batch_size):
        indices = np.random.choice(self.pt, batch_size, replace=False)
        return self.get_samples_recurrent(indices)
    
    def sample_sequences(self, ):
        pass

    def sample_episodes(self, ):
        pass
        
    def sample_batch(self, recurrent_seq_len: int,  whole_episodes: bool = False, batch_size: Optional[int] = 256) -> Generator[RolloutBufferSamples, None, None]:
        """
            Samples a batch of transitions for a policy update. If policy is recurrent
            then either samples sequences of length recurrent_seq_len or whole episodes.
            :param recurrent_seq_len: Number of timesteps to collect for a sequence from the starting point.
            :param whole_episodes: Whether to collect whole episodes for recurrent policy.
            :param batch_size: Number of timesteps if MLP otherwise number of sequences.
            :
        """
        if self.is_recurrent:
            if not whole_episodes:
                # Need to split all transitions into T x B x -1 to be fed to the recurrent layer
                # Permute the data gathered so far
                indices = np.random.permutation(self.buffer_size)
                
                start_idx = 0
                while start_idx < self.buffer_size:
                    # 1. Sample random batch_size of sequences
                    # 그냥 섞어버리면 시퀀스의 순서가 유지되지 않을텐데 어떻게 처리할까?
                    sequences = indices[start_idx:start_idx + batch_size]
                    # 2. For those episodes find the index in self.trajectory_index where they end
                    
                    episode_ends_idxs = np.digitize(sequences, bins=self.trajectory_index)
                    timesteps_eps_end = self.trajectory_index[episode_ends_idxs] - 1
                        # 3. For each sequence we need to collect recurrent_seq_len timesteps
                    # if not enough then pad.
                    observations, actions, old_values, returns, old_log_probs, advantages, masks, actor_hxs, critic_hxs  = [], [], [], [], [], [], [], [], []
                    for i, sequence in enumerate(sequences):
                        num_steps_to_end = (timesteps_eps_end[i] - sequence) + 1
                        # ex1: num_steps = 10, seq_len 16 -> 10 -> need padding?
                        # ex2: num_steps = 20, seq_len 16 -> 16
                        seq_len = num_steps_to_end if num_steps_to_end < recurrent_seq_len else recurrent_seq_len
                        samples = asdict(self._get_samples_recurrent(np.arange(sequence, sequence + seq_len)))
                        if seq_len < recurrent_seq_len: # 10 < 16
                            pad_len = recurrent_seq_len - seq_len # 6 pad length
                            for key, value in samples.items():
                                # if obs is image value.shape = [L, C, W, H] 
                                # value.shape[1:] = [C, W, H]
                                value = torch.cat((value, torch.zeros(((pad_len,) + value.shape[1:]), dtype=value.dtype).to(self.device)))
                                samples[key] = value
                        
                        observations.append(samples["observations"])
                        actions.append(samples["actions"])
                        old_values.append(samples["old_values"])
                        returns.append(samples["returns"])
                        old_log_probs.append(samples["old_log_probs"])
                        advantages.append(samples["advantages"])
                        masks.append(samples["masks"])
                        actor_hxs.append(samples["actor_hxs"])
                        critic_hxs.append(samples["critic_hxs"])

                    observations = torch.stack(observations, dim=1)
                    actions = torch.stack(actions, dim=1).view(-1, *self.action_dim)
                    old_values = torch.stack(old_values, dim=1).view(-1)
                    returns = torch.stack(returns, dim=1).view(-1)
                    old_log_probs = torch.stack(old_log_probs, dim=1).view(-1)
                    advantages = torch.stack(advantages, dim=1).view(-1)
                    masks = torch.stack(masks, dim=1).view(-1)
                    actor_hxs = torch.stack(actor_hxs, dim=1)
                    critic_hxs = torch.stack(critic_hxs, dim=1)

                    yield RecurrentRolloutBufferSamples(*tuple([observations, \
                                                        actions, old_values, \
                                                            returns, old_log_probs, \
                                                                advantages, masks, \
                                                                    actor_hxs, critic_hxs]))
                    start_idx += batch_size
            else:
                # truncated_batch는 무슨 의미일까?
                # buffer_size가 128이고 배치 사이즈가 3이라고 하자
                # trajectory_index = [0, 32, 64, 96, 128]
                truncated_batch = len(self.trajectory_index) % batch_size
                # Need to always keep a multiple of batch size + 1 trajectories
                # so that we can index the last trajectory's final timestep
                # If num episodes % batch_size == 0 then remove batch_size - 1 trajectories
                if truncated_batch == 0:
                    self.trajectory_index = self.trajectory_index[:-(batch_size - 1)]
                # If num episodes % batch_size == 1 then its fine nothing needs to be changed
                # If num episodes % batch_size > 1 then remove truncated_batch 
                
                # traj_index = np.array([0, 32, 64, 96, 128])
                # traj_idx = traj_index[:-(2 - (2-1))] = array([ 0, 32, 64, 96])
                # 버퍼에 전체 에피소드가 들어있기 때문에 마지막 traj index를 날린다 
                if truncated_batch > 1:
                    self.trajectory_index = self.trajectory_index[:-(truncated_batch - (truncated_batch - 1))]

                traj_indices = np.random.permutation(len(self.trajectory_index) - 1)
                start_idx = 0
                while start_idx < len(traj_indices):
                    batch_idx = traj_indices[start_idx:start_idx+batch_size]
                    observations = [to_tensor(self.observations[self.trajectory_index[idx]:self.trajectory_index[idx + 1]], device=self.device) for idx in batch_idx]
                    actions = [to_tensor(self.actions[self.trajectory_index[idx]:self.trajectory_index[idx + 1]], device=self.device) for idx in batch_idx]
                    old_values = [to_tensor(self.values[self.trajectory_index[idx]:self.trajectory_index[idx + 1]], device=self.device) for idx in batch_idx]
                    returns = [to_tensor(self.returns[self.trajectory_index[idx]:self.trajectory_index[idx + 1]], device=self.device) for idx in batch_idx]
                    old_log_probs = [to_tensor(self.log_probs[self.trajectory_index[idx]:self.trajectory_index[idx + 1]], device=self.device) for idx in batch_idx]
                    advantages = [to_tensor(self.advantages[self.trajectory_index[idx]:self.trajectory_index[idx + 1]], device=self.device) for idx in batch_idx]
                    actor_hxs = [to_tensor(self.actor_hxs[self.trajectory_index[idx]:self.trajectory_index[idx + 1]], device=self.device) for idx in batch_idx]
                    critic_hxs = [to_tensor(self.critic_hxs[self.trajectory_index[idx]:self.trajectory_index[idx + 1]], device=self.device) for idx in batch_idx]
                    masks = [torch.ones_like(r) for r in returns]

                    observations = pad_sequence(observations, batch_first=False)
                    actions = pad_sequence(actions, batch_first=False).view(-1, *self.action_dim)
                    old_values = pad_sequence(old_values, batch_first=False).view(-1)
                    returns = pad_sequence(returns, batch_first=False).view(-1)
                    old_log_probs = pad_sequence(old_log_probs, batch_first=False).view(-1)
                    advantages = pad_sequence(advantages, batch_first=False).view(-1)
                    masks = pad_sequence(masks, batch_first=False).view(-1)
                    actor_hxs = pad_sequence(actor_hxs, batch_first=False)
                    critic_hxs = pad_sequence(critic_hxs, batch_first=False)
                    yield RecurrentRolloutBufferSamples(*tuple([observations, actions, old_values, returns, old_log_probs, advantages, masks, actor_hxs, actor_cxs, critic_hxs, critic_cxs]))
                    start_idx += batch_size
        else:
            indices = np.random.permutation(self.buffer_size)
            start_idx = 0
            while start_idx < self.buffer_size:
                yield self._get_samples(indices[start_idx:start_idx + batch_size])
                start_idx += batch_size
                
