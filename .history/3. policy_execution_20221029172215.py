import logging
import gym
from gym.spaces import Box, Discrete
from distutils.util import strtobool
import os
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
import pygame
import argparse
from point_env import *
from ppornn import *
from rnn_policy import RNNCritic, RNNPolicy
from refactored_rolloutbuffer import RolloutBuffer
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# ! Environments
# env = SparsePointEnv(goal_radius=0.5, max_episode_steps=100, 
#                      goal_sampler='semi-circle', is_render=True)
env = gym.make('CartPole-v1')
#env = gym.make("LunarLanderContinuous-v2")
print(type(env))
if isinstance(env, SparsePointEnv):
    render_gym = False
else:
    render_gym = True

print(env.observation_space)
print(env.action_space)
state_dim = env.observation_space.shape[0]

if isinstance(env.action_space, Box):
    action_dim = env.action_space.shape[0]
    is_continuous = True

elif isinstance(env.action_space, Discrete):
    action_dim = env.action_space.n
    is_continuous = False


print(state_dim)
print(action_dim)

def parse_args():
    parser = argparse.ArgumentParser()

    # ? 1. Experiments parameters, Environments information
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)),
                        default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--env-id", type=str, default="CartPole-v0",
            help="the id of the environment")
    parser.add_argument('--is_continuous', default=is_continuous)
    parser.add_argument("--seed", type=int, default=1,
            help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
            help="total timesteps of the experiments")
    parser.add_argument("--num-envs", type=int, default=1,
            help="the number of parallel game environments")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
            help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--device', default='cuda')


    # ? 3. Networks, Activation, Hyperparameters
    parser.add_argument('--lr', default=0.0001)
    parser.add_argument("--batch_size", type=int, default=16,
            help="the size of minibatch")
    # parser.add_argument("--num-minibatches", type=int, default=32,
    #         help="the number of mini-batches")
    parser.add_argument('--rollout_steps', default=128)
    parser.add_argument("--update-epochs", type=int, default=5,
            help="the K epochs to update the policy")
    parser.add_argument('--state_dim', default=state_dim)
    parser.add_argument('--linear_dim', default=128)
    parser.add_argument('--hidden_dim', default=32)
    parser.add_argument('--seq_len', type=int, default=4)
    parser.add_argument('--is_recurrent', type=bool, default=True)
    parser.add_argument('--num_rnn_layers', default=2)
    parser.add_argument('--action_dim', default=action_dim)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--gae_lambda', default=0.95)
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")

    args = parser.parse_args()
    return args


def get_init_rnn_state(num_rnn_layers, hidden_dim, batch_size, is_training, device):
    if is_training:
        hidden = torch.zeros(num_rnn_layers, batch_size, hidden_dim).to(device)
    else:
        hidden = torch.zeros(num_rnn_layers, 1, hidden_dim).to(device)
    return hidden        


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    # ? 3. Logger

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    
    goal = semi_circle_goal_sampler()
    logger.debug(goal)
    rnn_policy = RNNPolicy(args).to(device)
    rnn_critic = RNNCritic(args).to(device)
    rollout_buffer = RolloutBuffer(buffer_size=args.rollout_steps,
                                state_dim=env.observation_space.shape,
                                action_space=env.action_space,
                                gamma=args.gamma,
                                gae_lambda=args.gae_lambda,
                                device="cuda",
                                is_recurrent=True,
                                recurrent_size=args.hidden_dim,
                                num_rnn_layers=args.num_rnn_layers)
    
    policy_optimizer = optim.Adam(rnn_policy.parameters(), lr=args.lr, eps=1e-5)
    critic_optimizer = optim.Adam(rnn_critic.parameters(), lr=args.lr, eps=1e-5)

    global_step = 0
    for e in range(10):
        done = False
        obs = env.reset()
        step = 0
        total_rewards = 0
        policy_hidden = get_init_rnn_state(args.num_rnn_layers, args.hidden_dim, \
                                        batch_size=1, device=device, is_training=False)
        critic_hidden = get_init_rnn_state(args.num_rnn_layers, args.hidden_dim, \
                                        batch_size=1, device=device, is_training=False)
        while not done and step < 500:
            for i in range(args.rollout_steps):
                global_step += 1
                step += 1
                
                if render_gym:
                    env.render()
                else:
                    env.render(mode='rgb', tick=0.001)
                    env.render(mode='text', tick=None)
                
                with torch.no_grad():
                    action, log_prob, policy_hidden = rnn_policy.choose_action(obs, policy_hidden)
                    value, critic_hidden = rnn_critic(obs, critic_hidden)
                print(f"policy hidden: {policy_hidden}")
                print(f"critic hidden: {critic_hidden}")
                next_obs, reward, done, info = env.step(action)
                total_rewards += reward
                sample = (obs, action, value, reward, log_prob,
                        done, policy_hidden, critic_hidden)
                rollout_buffer.add_sample(sample)
                obs = next_obs
                print(rollout_buffer.pt)
                
                
                if done or step == 500:
                    break
                
            #train_ppo(args, rnn_policy, rnn_critic, rollout_buffer, policy_optimizer, critic_optimizer, writer, global_step)
            
            if done or step == 500:
                rollout_buffer.end_trajectory(value, done)
                writer.add_scalar("charts/total_rewards", total_rewards, global_step)
                break
                
    
