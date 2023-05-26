'''
INSTRUCTIONS:
=============
This algorithm is a derivation from CleanRL's MultiDescrete PPO script
https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_multidiscrete.py

1. Install dependencies:
	`pip install torch gymnasium scipy git+https://github.com/sarisabban/Pose`

2. To train an agent (training time 30 days):
	`python3 -B RL.py -rl`

3. To play the environment using a trained agent:
	`python3 -B RL.py -rlp agent.pth`

The following is a SLURM and PBS job submission scripts to train the agent on a high performance supercomputer:
----------------------------
#!/bin/sh
#SBATCH --job-name=Mira
#SBATCH --partition=compsci
#SBATCH --time=720:00:00
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48

cd $SLURM_SUBMIT_DIR

python3 -u -B RL.py -rl
----------------------------
#!/bin/bash
#PBS -N Mira
#PBS -q thin
#PBS -l walltime=720:00:00
#PBS -l select=1:ncpus=24
#PBS -j oe

cd $PBS_O_WORKDIR

conda activate RL
python3 -u -B RL.py -rl
----------------------------
'''

import sys
import time
import torch
import random
import argparse
import warnings
import datetime
import numpy as np
import gymnasium as gym
from MiraMar import MiraMar
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Reinforcement learning training on the MiraMar environment')
parser.add_argument('-rl', '--rl_train', action='store_true', help='Train a reinforcement learning agent')
parser.add_argument('-rlp', '--rl_play', nargs='+', help='Have a trained agent play the game using the agent.pth file')
args = parser.parse_args()

def make_env(env_id):
	def thunk():
		env = env_id
		return env
	return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
	torch.nn.init.orthogonal_(layer.weight, std)
	torch.nn.init.constant_(layer.bias, bias_const)
	return layer

class Agent(torch.nn.Module):
	def __init__(self, envs):
		super(Agent, self).__init__()
		obs_shape = envs.single_observation_space.shape
		self.network = torch.nn.Sequential(
			layer_init(torch.nn.Linear(np.array(obs_shape).prod(), 64)),
			torch.nn.ReLU(),
			layer_init(torch.nn.Linear(64, 64)),
			torch.nn.ReLU(),
			layer_init(torch.nn.Linear(64, 128)),
			torch.nn.ReLU(),)
		self.nvec = envs.single_action_space.nvec
		self.actor = layer_init(torch.nn.Linear(128, self.nvec.sum()), std=0.01)
		self.critic = layer_init(torch.nn.Linear(128, 1), std=1)
	def get_value(self, x):
		return self.critic(self.network(x))
	def get_action_and_value(self, x, action=None):
		hidden = self.network(x)
		logits = self.actor(hidden)
		split_logits = torch.split(logits, self.nvec.tolist(), dim=1)
		multi_categoricals = [torch.distributions.categorical.Categorical(logits=logits) for logits in split_logits]
		if action is None: action = torch.stack([categorical.sample() for categorical in multi_categoricals])
		logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
		entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
		return action.T, logprob.sum(0), entropy.sum(0), self.critic(hidden)

def train():
	''' Train a PPO agent on the MiraMar environment '''
	# Define variables
	env           = MiraMar()
	n_envs        = 64
	n_steps       = 1024
	timesteps     = 144e6
	n_minibatches = 128
	epochs        = 16
	seed          = 1
	lr            = 2.5e-4
	gamma         = 0.99
	lambd         = 0.95
	clip_coef     = 0.1
	vf_coef       = 0.5
	ent_coef      = 0.01
	max_grad_norm = 0.5
	target_kl     = 0.015
	log           = True
	# Fix seeds to make all experiments reproducible
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	date = datetime.datetime.now().strftime('%d-%b-%Y @ %H:%M:%S')
	print('Training on:', device, '||', 'Started on:', date, '\n' + '='*40)
	# Environment setup
	envs = gym.vector.AsyncVectorEnv([make_env(env) for i in range(n_envs)])
	agent = Agent(envs).to(device)
	optimizer = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
	batch_size = int(n_envs * n_steps)
	minibatch_size = int(batch_size // n_minibatches)
	n_updates = int(timesteps // batch_size)
	# Storage buffer setup
	s_obs_space = envs.single_observation_space.shape
	s_act_space = envs.single_action_space.shape
	obs         = torch.zeros((n_steps, n_envs) + s_obs_space).to(device)
	actions     = torch.zeros((n_steps, n_envs) + s_act_space).to(device)
	rewards     = torch.zeros((n_steps, n_envs)).to(device)
	values      = torch.zeros((n_steps, n_envs)).to(device)
	dones       = torch.zeros((n_steps, n_envs)).to(device)
	logprobs    = torch.zeros((n_steps, n_envs)).to(device)
	# Start the environment
	next_obs    = torch.Tensor(envs.reset(seed=seed)[0]).to(device)
	next_done   = torch.zeros(n_envs).to(device)
	# Updates
	global_step = 0
	for update in range(1, n_updates + 1):
		# Anneal the learning rate
		time_start = time.time()
		optimizer.param_groups[0]['lr'] = (1.0 - (update - 1.0) / n_updates) * lr
		# Steps to generate a dataset
		Gts = []
		for step in range(n_steps):
			global_step += 1 * n_envs
			obs[step] = next_obs
			dones[step] = next_done
			# Take action
			with torch.no_grad():
				action, logprob, _, value = agent.get_action_and_value(next_obs)
				values[step] = value.flatten()
			actions[step] = action
			logprobs[step] = logprob
			# Play game using action
			next_obs, reward, term, trun, info = envs.step(action.cpu().numpy())
			done = term + trun
			index = np.where(done==True)[0]
			rewards[step] = torch.tensor(reward).to(device).view(-1)
			next_obs  = torch.Tensor(next_obs).to(device)
			next_done = torch.Tensor(done).to(device)
			if 'final_info' in info.keys():
				for e in info['final_info'][index]:
					Gt = round(e['episode']['r'], 3)
					Gts.append(Gt)
		# Bootstrap value using GAE
		with torch.no_grad():
			next_value = agent.get_value(next_obs).reshape(1, -1)
			advantages = torch.zeros_like(rewards).to(device)
			lastgaelam = 0
			for t in reversed(range(n_steps)):
				if t == n_steps - 1:
					nextnonterminal = 1.0 - next_done
					nextvalues = next_value
				else:
					nextnonterminal = 1.0 - dones[t + 1]
					nextvalues = values[t + 1]
				delta = rewards[t] + gamma * nextvalues * nextnonterminal-values[t]
				advantages[t] = lastgaelam = delta + gamma * lambd * nextnonterminal * lastgaelam
			returns = advantages + values
		# Flatten the dataset
		b_obs        = obs.reshape((-1,) + envs.single_observation_space.shape)
		b_actions    = actions.reshape((-1,) + envs.single_action_space.shape)
		b_returns    = returns.reshape(-1)
		b_advantages = advantages.reshape(-1)
		b_values     = values.reshape(-1)
		b_logprobs   = logprobs.reshape(-1)
		b_indx       = np.arange(batch_size)
		# Optimise the policy (Actor) and value (Critic) neural networks
		clipfracs = []
		for epoch in range(epochs):
			np.random.shuffle(b_indx)
			for start in range(0, batch_size, minibatch_size):
				# Get minibatch
				end = start + minibatch_size
				mb_inds = b_indx[start:end]
				# Push minibatch through Actor/Critic networks 
				_, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds].T)
				logratio = newlogprob - b_logprobs[mb_inds]
				ratio = logratio.exp()
				# Calculate Advantage
				mb_adv = b_advantages[mb_inds]
				mb_advantages = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
				# Policy loss
				pg_loss1 = -mb_advantages * ratio
				pg_loss2 = -mb_advantages * torch.clamp(ratio, 1-clip_coef, 1+clip_coef)
				pg_loss = torch.max(pg_loss1, pg_loss2).mean()
				# Value loss
				newvalue = newvalue.view(-1)
				v_loss_unclipped = (newvalue - b_returns[mb_inds])**2
				v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], clip_coef, clip_coef,)
				v_loss_clipped = (v_clipped - b_returns[mb_inds])**2
				v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
				v_loss = 0.5 * v_loss_max.mean()
				# Entropy loss
				entropy_loss = entropy.mean()
				# Final loss
				loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
				# Backpropagation & gradient descent
				optimizer.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
				optimizer.step()
				# Aproximate KL divergence
				with torch.no_grad():
					approx_kl = ((ratio - 1) - logratio).mean()
					clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]
			if target_kl is not None:
				if approx_kl > target_kl: break
		# Explained variance
		y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
		var_y = np.var(y_true)
		explained_var = np.nan if var_y == 0 else 1-np.var(y_true - y_pred) / var_y
		# Time keeping
		time_seconds = time.time() - time_start
		time_update_seconds = round(time_seconds * (n_updates - update), 0)
		time_update = datetime.timedelta(seconds=time_update_seconds)
		time_seconds = round(time_seconds, 3)
		if log:
			with open('train.log', 'a') as f:
				Gt_mean = round(np.array(Gts).mean(), 3)
				Gt_SD   = round(np.array(Gts).std(), 3)
				A_loss  = round(pg_loss.item(), 3)
				C_loss  = round(v_loss.item(), 3)
				Entropy = round(entropy_loss.item(), 3)
				Loss    = round(loss.item(), 3)
				KL      = round(approx_kl.item(), 3)
				Clip    = round(clipfracs[-1], 3)
				exp_var = round(explained_var, 3)
				A = f'Update: {update:,}/{n_updates:<10,}'
				B = f'Steps: {global_step:<10,}'
				C = f'Returns: {Gt_mean:,} +- {Gt_SD:<10,}'
				D = f'A_loss: {A_loss:<10,}'
				E = f'C_loss: {C_loss:<10,}'
				F = f'Entropy loss: {Entropy:<10,}'
				G = f'Final loss: {Loss:<10,}'
				H = f'KL: {KL:<10,}'
				I = f'Clip: {Clip:<10,}'
				J = f'Explained Variance: {exp_var:<10,}'
				K = f'Time per update: {time_seconds:<10,}s'
				L = f'Remaining time: {time_update}\n'
				f.write(A + B + C + D + E + F + G + H + I + J + K + L)
			# Export agent model every 100 updates
			if (update % 50 == 0): 
				# Export agent model
				torch.save(agent, f'agent_{update}.pth')
		print(f'Updates: {update}/{n_updates} | Steps: {global_step:<10,} Return: {Gt_mean:,} +- {Gt_SD:<15,} Remaining time: {time_update}')
	# Export agent model
	torch.save(agent, 'agent.pth')

def play(filename='agent.pth'):
	''' Play the MiraMar environment using a trained PPO agent '''
	# Import agent model
	agent = torch.load(filename)
	agent.eval()
	# Play game
	env = MiraMar()
	S, I = env.reset()
	done = False
	while not done:
		S = torch.Tensor([S]).to('cpu')
		A, _, _, _ = agent.get_action_and_value(S)
		S, R, T, U, I = env.step(A[0].numpy())
		done = bool(T or U)
	print('Actions:', I['actions'])
	print('Rewards:', I['rewards'])
	print('Episode:', I['episode'])
	env.render()

def main():
	if args.rl_train:  train()
	elif args.rl_play: play(filename=sys.argv[2])

if __name__ == '__main__': main()
