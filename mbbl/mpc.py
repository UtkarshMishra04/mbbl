import numpy as np

#========================================================
# 
# Cost function for a whole trajectory:
#

def trajectory_cost_fn(cost_fn, states, actions, next_states):
	trajectory_cost = 0
	for i in range(len(actions)):
		trajectory_cost += cost_fn(states[i], actions[i], next_states[i])
	return trajectory_cost

#========================================================
# 
# DMDMPC_RL Controller
#

class DMDMPC_RLcontroller():
	
	""" Controller built using the MPC method outlined in Online Learning Approach to MPC & ARS """
	def __init__(self, 
				 env, 
				 dyn_model, 
				 horizon=20, 
				 cost_fn=None, 
				 num_simulated_paths=100,
				 percent_elite = 5,
				 gamma = 0.9
				 ):
		self.env = env
		self.dyn_model = dyn_model
		self.horizon = horizon
		self.cost_fn = cost_fn
		self.num_simulated_paths = num_simulated_paths
		self.percent_elite = percent_elite
		self.gamma = gamma
		#self.mean = np.full((horizon,env.action_space.shape[0]),0)
		self.std = 0.1*np.identity((6))
		self.num_elite = self.num_simulated_paths*self.percent_elite

	def get_action(self, state, rl_policy):

		obs, obs_list, obs_next_list, act_list = [], [], [], []
		[obs.append(state) for _ in range(self.num_simulated_paths)]
		
		for step in range(self.horizon):
			
			obs_list.append(obs)
			actions = []
			
			for j in range(self.num_simulated_paths):
				
				#Get Mean to sample actions from existing RL policy
				rl_action_mean = rl_policy(obs[j])  
				rl_action_mean = np.clip(rl_action_mean, 0.99*self.env.action_space.low, 0.99*self.env.action_space.high)

				#sample actions from RL actions as mean and add noise
				action = np.random.multivariate_normal(rl_action_mean,self.std)
				action += np.random.randn(self.env.action_space.shape[0])

				#clip the action 
				action = np.clip(action, 0.99*self.env.action_space.low, 0.99*self.env.action_space.high)	
				actions.append(action)
			act_list.append(actions)
			obs = self.dyn_model.predict(np.array(obs), np.array(actions))
			obs_next_list.append(obs)

		#get the tragectory cost for all trajectories
		trajectory_cost_list = trajectory_cost_fn(self.cost_fn, np.array(obs_list), np.array(act_list), np.array(obs_next_list)) 

		#get the elite trajectories 
		elite_inds = trajectory_cost_list.argsort()[0:self.num_elite]
				
		#weight the actions with costs in elite indices
		act_list=np.array(act_list)
		weighted_actions=np.array([act_list[:,i,:]*trajectory_cost_list[i] for i in elite_inds])
		
		#calculate the gradient to update mean (wighted actions/sum of weights)
		grad_mean=np.sum(weigted_actions, axis=0)
		total_cost=np.sum(trajectory_cost_list[elite_inds])
		grad_mean=grad_mean/total_cost

		#Get the mean for first step
		action_mean=(1-self.gamma)*rl_policy(state)+self.gamma*grad_mean[0]

		#apply the mean directly or sample from the mean
		action=np.clip(action_mean, self.env.action_space.low, self.env.action_space.high)
		
		return action
