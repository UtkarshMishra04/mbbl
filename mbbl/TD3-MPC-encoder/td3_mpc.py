#!/usr/bin/env python
import pickle
import random
import numpy as np
import torch
import torch.optim as optim

from Policy_ddpg import Policy
from Value_ddpg import Value
from td3_step import td3_step
from replay_memory import Memory
from env_util import get_env_info
from file_util import check_path
from torch_util import device, FLOAT
from zfilter import ZFilter

#MPC Controller & Environmenrt specific Cost Function
from mpc_controllers import DMDMPCcontroller, RandomController
from cost_functions import trajectory_cost_fn, cheetah_cost_fn

from dynamics import Encoder

class TD3_MPC:
    def __init__(self,
                 env_id,
                 dim_latent,
                 render=False,
                 num_process=1,
                 memory_size=1000000,
                 lr_p=1e-3,
                 lr_v=1e-3,
                 discount_factor=0.99,
                 polyak=0.995,
                 action_noise=0.1,
                 target_action_noise_std=0.2,
                 target_action_noise_clip=0.5,
                 explore_size=10000,
                 step_per_iter=3000,
                 batch_size=100,
                 min_update_step=1000,
                 update_step=50,
                 policy_update_delay=2,
                 seed=1,
                 model_path=None,
                 planning_h=10,
                 simulated_paths=100,
                 elite_fraction=0.5,
                 alpha=0.5,
                 gamma=0.8
                 ):
        self.env_id = env_id
        self.dim_latent = dim_latent
        self.discount_factor= discount_factor
        self.polyak = polyak
        self.action_noise = action_noise
        self.target_action_noise_std = target_action_noise_std
        self.target_action_noise_clip = target_action_noise_clip
        self.memory = Memory(memory_size)
        self.explore_size = explore_size
        self.step_per_iter = step_per_iter
        self.render = render
        self.num_process = num_process
        self.lr_p = lr_p
        self.lr_v = lr_v
        self.batch_size = batch_size
        self.min_update_step = min_update_step
        self.update_step = update_step
        self.policy_update_delay = policy_update_delay
        self.model_path = model_path
        self.seed = seed
        self.gamma=gamma
        self.alpha=alpha

        self._init_model()

        cost_fn = cheetah_cost_fn #env_name_cost_fn
        self.mpc_controller = DMDMPCcontroller(env_id=env_id, 
                                    dynamics = self.encodings,
                                    horizon=planning_h, 
                                    cost_fn=cost_fn, 
                                    num_simulated_paths=simulated_paths,
                                    elite_fraction=elite_fraction,
                                    gamma=gamma,
                                    alpha=alpha)

        

    def _init_model(self):
        """init model from parameters"""
        self.env, env_continuous, num_states, self.num_actions = get_env_info(
            self.env_id)
        assert env_continuous, "TD3 is only applicable to continuous environment !!!!"

        self.action_low, self.action_high = self.env.action_space.low[
            0], self.env.action_space.high[0]
        # seeding
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.env.seed(self.seed)

        self.policy_net = Policy(
            num_states, self.num_actions, self.action_high).to(device)
        self.policy_net_target = Policy(
            num_states, self.num_actions, self.action_high).to(device)

        self.value_net_1 = Value(num_states, self.num_actions).to(device)
        self.value_net_target_1 = Value(
            num_states, self.num_actions).to(device)
        self.value_net_2 = Value(num_states, self.num_actions).to(device)
        self.value_net_target_2 = Value(
            num_states, self.num_actions).to(device)

        self.running_state = ZFilter((num_states,), clip=5)

        self.num_states = num_states
        self.encodings = Encoder(self.num_states, self.dim_latent, self.num_actions)



        if self.model_path:
            print("Loading Saved Model {}_td3.p".format(self.env_id))
            self.policy_net, self.value_net_1, self.value_net_2, self.running_state = pickle.load(
                open('{}/{}_td3.p'.format(self.model_path, self.env_id), "rb"))

        self.policy_net_target.load_state_dict(self.policy_net.state_dict())
        self.value_net_target_1.load_state_dict(self.value_net_1.state_dict())
        self.value_net_target_2.load_state_dict(self.value_net_2.state_dict())

        self.optimizer_p = optim.Adam(
            self.policy_net.parameters(), lr=self.lr_p)
        self.optimizer_v_1 = optim.Adam(
            self.value_net_1.parameters(), lr=self.lr_v)
        self.optimizer_v_2 = optim.Adam(
            self.value_net_2.parameters(), lr=self.lr_v)

    def choose_action(self, state, noise_scale):
        """select action"""
        state = FLOAT(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob = self.policy_net.get_action_log_prob(state)
        action = action.cpu().numpy()[0]
        # add noise
        noise = noise_scale * np.random.randn(self.num_actions)
        action += noise
        action = np.clip(action, -self.action_high, self.action_high)
        return action 
    
    def choose_action_mpc(self, state, noise_scale):
        """select action"""
        state = FLOAT(state).to(device)
        with torch.no_grad():
            action_rl, log_prob = self.policy_net.get_action_log_prob(state)
        action_mpc = self.mpc_controller.get_control(state, action_rl, self.alpha)
        # add noise
        noise = noise_scale * np.random.randn(self.num_actions)
        action_mpc += noise
        action_mpc = np.clip(action_mpc, -self.action_high, self.action_high)
        
        return action_rl, action_mpc

    def eval(self, i_iter, render=False):
        """evaluate model"""

        state = self.env.reset()
        test_reward = 0
        while True:
            if render:
                self.env.render()
            action_rl, action_mpc = self.choose_action_mpc(state, 0)
            action = action_mpc
            print("action in eval", action)
            state, reward, done, _ = self.env.step(action)

            test_reward += reward
            if done:
                break
        print(f"Iter: {i_iter}, test Reward: {test_reward}")
        self.env.close()

    def learn(self, writer, i_iter):
        """interact"""
        global_steps = (i_iter - 1) * self.step_per_iter
        log = dict()
        num_steps = 0
        num_episodes = 0
        total_reward = 0
        min_episode_reward = float('inf')
        max_episode_reward = float('-inf')

        while num_steps < self.step_per_iter:
            state = self.env.reset()
            episode_reward = 0

            for t in range(10000):

                if self.render:
                    self.env.render()

                if global_steps < self.explore_size:  # explore
                    action = self.env.action_space.sample()
                else:  # action with noise
                    action_rl, action_mpc = self.choose_action_mpc(state, self.action_noise)
                    #print("rl action:",action_rl, type(action_rl))
                    #print("mpc action:",action_mpc, type(action_mpc))
                    action = action_mpc
                next_state, reward, done, _ = self.env.step(action)
                # next_state = self.running_state(next_state)
                mask = 0 if done else 1
                # ('state', 'action', 'reward', 'next_state', 'mask', 'log_prob')
                self.memory.push(state, action, reward, next_state, mask, None)

                episode_reward += reward
                global_steps += 1
                num_steps += 1

                if global_steps >= self.min_update_step and global_steps % self.update_step == 0:
                    for k in range(self.update_step):
                        batch, permuted_batch = self.memory.sample(self.batch_size)  # random sample batch
                        self.update(batch, permuted_batch, k)

                if done or num_steps >= self.step_per_iter:
                    break

                state = next_state

            num_episodes += 1
            total_reward += episode_reward
            min_episode_reward = min(episode_reward, min_episode_reward)
            max_episode_reward = max(episode_reward, max_episode_reward)

        self.env.close()

        log['num_steps'] = num_steps
        log['num_episodes'] = num_episodes
        log['total_reward'] = total_reward
        log['avg_reward'] = total_reward / num_episodes
        log['max_episode_reward'] = max_episode_reward
        log['min_episode_reward'] = min_episode_reward

        print(f"Iter: {i_iter}, num steps: {log['num_steps']}, total reward: {log['total_reward']: .4f}, "
              f"min reward: {log['min_episode_reward']: .4f}, max reward: {log['max_episode_reward']: .4f}, "
              f"average reward: {log['avg_reward']: .4f}")

        # record reward information
        writer.add_scalar("rewards/total_reward", log['total_reward'], i_iter)
        writer.add_scalar("rewards/average_reward", log['avg_reward'], i_iter)
        writer.add_scalar("rewards/min_reward", log['min_episode_reward'], i_iter)
        writer.add_scalar("rewards/max_reward", log['max_episode_reward'], i_iter)
        writer.add_scalar("rewards/num_steps", log['num_steps'], i_iter)

        writer = self.encodings.update_writer(writer, i_iter)
        
    def update(self, batch, batch2, k_iter):
        """learn model"""
        batch_state = FLOAT(batch.state).to(device)
        batch_action = FLOAT(batch.action).to(device)
        batch_reward = FLOAT(batch.reward).to(device)
        batch_next_state = FLOAT(batch.next_state).to(device)
        batch_mask = FLOAT(batch.mask).to(device)

        # update by TD3
        alg_step_stats = td3_step(self.policy_net, self.policy_net_target, self.value_net_1, self.value_net_target_1, self.value_net_2,
                                  self.value_net_target_2, self.optimizer_p, self.optimizer_v_1, self.optimizer_v_2, batch_state,
                                  batch_action, batch_reward, batch_next_state, batch_mask, self.gamma, self.polyak,
                                  self.target_action_noise_std, self.target_action_noise_clip, self.action_high,
                                  k_iter % self.policy_update_delay == 0)

        batch_state2 = FLOAT(batch2.state).to(device)
        batch_action2 = FLOAT(batch2.action).to(device)
        batch_reward2 = FLOAT(batch2.reward).to(device)
        batch_next_state2 = FLOAT(batch2.next_state).to(device)
        batch_mask2 = FLOAT(batch2.mask).to(device)

        self.encodings.update_encoder(batch_state, batch_action, batch_reward, batch_next_state, 
                                    batch_state2, batch_action2, batch_reward2, batch_next_state2)

    def save(self, save_path):
        """save model"""
        check_path(save_path)
        pickle.dump((self.policy_net, self.value_net_1, self.value_net_2, self.running_state),
                    open('{}/{}_td3.p'.format(save_path, self.env_id), 'wb'))
