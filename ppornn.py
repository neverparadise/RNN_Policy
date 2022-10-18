import torch
import torch.nn.functional as F

import numpy as np

def train_ppo(args, policy, critic, rollout_buffer, policy_opt, value_opt, writer, global_step):
    if len(rollout_buffer.trajectory_index) < 2:
        return
    
    if args.is_recurrent:
        for k in range(args.update_epochs):
            for sample in rollout_buffer.sample_batch(recurrent_seq_len=args.seq_len ,whole_episodes=False, batch_size=args.batch_size):
                observations, actions, values, returns, log_probs, advantages, masks, actor_hxs, critic_hxs = sample.get_transition()
            
            # ! obs, hidden state 문제 있는듯
            # [seq_len, batch_size, state_size]
            
                policy_loss, entropy, approx_kl_divs, clip_frac = compute_policy_loss(args, policy, observations, actions, log_probs, advantages, actor_hxs, masks)
                critic_loss = compute_critic_loss(critic, observations, returns, critic_hxs, masks)
                loss = policy_loss + args.vf_coef * critic_loss
                policy_opt.zero_grad()
                value_opt.zero_grad()
                loss.backward()
                policy_opt.step()
                value_opt.step()
        writer.add_scalar("losses/critic_loss", critic_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl_divs.item(), global_step)
        writer.add_scalar("losses/clip_frac", np.mean(clip_frac), global_step)
        
    else:
        observations, actions, values, returns, log_probs, advantages = rollout_buffer.sample_batch(recurrent_seq_len=args.seq_len
                                                                                                    ,whole_episodes=False,
                                                                                                    batch_size=args.batch_size)
        pass
        
def compute_policy_loss(args, policy, states: torch.Tensor, actions: torch.Tensor, old_log_probs: torch.Tensor, advantages: torch.Tensor, actor_hxs:torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
            Computes the actor policy loss.
            :param states: Environment states.
            :param actions: Actions performed by agent.
            :param old_log_probs: Log probability of actions of old policy.
            :param advantages: Advantages calculated with GAE. https://arxiv.org/abs/1506.02438
            :param masks: If a recurrent policy is used then this masks out padded sequences.
            :return: The actor policy loss.
        """
        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        curr_log_probs, entropy, actor_hidden = policy.choose_action(states, actor_hxs, is_training=True)
        ratios = torch.exp(curr_log_probs - old_log_probs)
        surr_one = ratios * advantages
        surr_two = torch.clamp(ratios, 1 - args.clip_coef, 1 + args.clip_coef) * advantages
        pg_loss = -torch.min(surr_one, surr_two)
        # https://github.com/DLR-RM/stable-baselines3/blob/df6f9de8f46509dad47e6d2e5620aa993b0fc883/stable_baselines3/ppo/ppo.py#L226-L230    
        if entropy is None:
            if masks is None:
                entropy = -torch.mean(-curr_log_probs)
            else:
                entropy = -torch.mean(-((curr_log_probs.T * masks).sum() / torch.clamp((torch.ones_like(curr_log_probs.T) * masks).float().sum(), min=1.0)))
        else:
            if masks is None:
                entropy = -torch.mean(entropy)
            else:
                entropy = -torch.mean((entropy.T * masks).sum() / torch.clamp((torch.ones_like(entropy.T) * masks).float().sum(), min=1.0))
            
        if masks is not None:
            pg_loss = (pg_loss.T * masks).sum() / torch.clamp((torch.ones_like(pg_loss.T) * masks).float().sum(), min=1.0)

        policy_loss = pg_loss.mean() + args.ent_coef * entropy
        
        approx_kl_divs = torch.mean((ratios - 1) - (curr_log_probs - old_log_probs))
        clip_frac = torch.mean((torch.abs(ratios - 1) > args.clip_coef).float())

        return policy_loss, entropy, approx_kl_divs, clip_frac

def compute_critic_loss(critic, states: torch.Tensor, returns: torch.Tensor, critic_hxs:torch.Tensor, masks: torch.Tensor = None) -> torch.Tensor:
    """
        Computes the critic loss.
        :param states: Environment states.
        :param returns: The discounted returns calculated by  advantages + state values.
        :param masks: If a recurrent policy is used then this masks out padded sequences.
        :return: The critic loss.
    """
    state_values, critic_hidden = critic.value(states, critic_hxs)
    if masks is None:
        critic_loss = F.mse_loss(returns, state_values.squeeze())
    else:
        critic_loss = (returns - state_values.squeeze()) ** 2
        critic_loss = (critic_loss.T * masks).sum() / torch.clamp((torch.ones_like(critic_loss.T) * masks).float().sum(), min=1.0)
        
    return critic_loss