import numpy as np
import copy
import time
import torch

from env_util import get_env_info
from cost_functions import trajectory_cost_fn
from torch_util import FLOAT, device

class RandomController():
	def __init__(self, env):
		self.env = env

	def get_action(self, state):
		return self.env.action_space.sample()

class DMDMPCcontroller():
	""" Controller built using the MPC method outlined in Online Learning Approach to MPC """
	def __init__(self, 
				 env_id, 
				 dynamics,
				 horizon=20, 
				 cost_fn=None, 
				 num_simulated_paths=1000,
				 elite_fraction=50,
				 gamma = 0.9,
				 alpha=0.5
				 ):
		self.env_id = env_id
		self.dynamics = dynamics	
		self.horizon = horizon
		self.cost_fn = cost_fn
		self.num_simulated_paths = num_simulated_paths
		self.gamma = gamma
		self.elite_fraction=elite_fraction
		self.alpha=alpha

		self.env, env_continuous, num_states, self.num_actions = get_env_info(self.env_id)
		self.mean = np.array([np.array([0.]*self.num_actions) for _ in range(horizon)]) #np.full((horizon,self.num_actions),0., dtype=float)
		self.std = 0.4*np.identity((self.num_actions))
		self.elite = self.num_simulated_paths*elite_fraction/100

		self.weighted_actions = self.grad_m = [0.]*self.horizon

	def get_control(self, state, rl_action, alpha):
		obs, obs_list, obs_next_list, act_list = [], [], [], [] 
		[obs.append(state) for _ in range(self.num_simulated_paths)]

		for step in range(self.horizon):

			obs_list.append(obs)

			actions = []
			
			for _ in range(self.num_simulated_paths):
				action = np.random.multivariate_normal(self.mean[step],self.std)
				action = np.clip(action, 0.99*self.env.action_space.low, 0.99*self.env.action_space.high)	
				actions.append(FLOAT(action).to(device))
			act_list.append(actions)

			#curr_states = FLOAT(obs).to(device)
			#controls = act_list.cpu().numpy()[0]

			with torch.no_grad():
				obs = self.dynamics.predict_state_trajectory(obs,actions)
				print(obs, type(obs))
			obs_next_list.append(obs)

		
		#trajectory_cost_list = trajectory_cost_fn(self.cost_fn, np.array(obs_list), np.array(act_list), np.array(obs_next_list)) 	
		
		#convert list of tensors from gpu to cpu numpy?		
		for i in range(self.horizon):
				for j in range(self.num_simulated_paths):
					if i == 0:obs_list[i][j] = obs_list[i][j].cpu().numpy()
					obs_next_list[i][j] = obs_next_list[i][j].cpu().numpy() 
					act_list[i][j] = act_list[i][j].cpu().numpy()

		
		trajectory_cost_list = np.array(trajectory_cost_fn(self.cost_fn, np.array(obs_list), np.array(act_list), np.array(obs_next_list)))
		#print(trajectory_cost_list)
		#print(self.elite, type(self.elite))
		elite_inds = trajectory_cost_list.argsort()[0:int(self.elite)]
		act_list=np.array(act_list)
		weighted_actions=np.array([act_list[:,i,:]*trajectory_cost_list[i] for i in elite_inds])
		
		#print(weighted_actions)
		grads = np.sum(weighted_actions, axis=0)
		
		#print("grads",grads)
		#compute the grad from elite fractions
		self.grad_m=grads/np.sum(trajectory_cost_list[elite_inds])
		
		#Firt control to Real System 

		grad_step = copy.deepcopy(self.grad_m[0])
		mean_action = (1-alpha)*rl_action.cpu().numpy()+alpha*grad_step
		control= np.clip(mean_action, self.env.action_space.low, self.env.action_space.high)

		#update other means
		#print("self.mean:",  np.array(self.mean))
		#print("self.grad_m:", self.grad_m)

		self.mean=(1-self.gamma)*self.mean+self.gamma*self.grad_m

		#shift means, std
		last_mean = self.mean[self.horizon-1].copy()
		self.mean=np.vstack((self.mean[1:],last_mean))
		

		return control
