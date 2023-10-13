'''
INSTRUCTIONS:
=============
This algorithm is a derivation from CleanRL's MultiDescrete PPO script
https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_multidiscrete.py

1. Install dependencies:
	`pip install torch gymnasium scipy git+https://github.com/sarisabban/Pose`

2. To train an agent (training time 30 days):
	`python3 -B RL.py -t`

3. To play the environment using a trained agent:
	`python3 -B RL.py -p agent.pth`

4. To generate a molecule for a custom path and targets:
	`python3 -B RL_6.py -g agent.pth Cx Cy Cz a b o j w T1x T1y T1z T2x T2y T2z ...`
	example:
	`python3 -B RL_6.py -g agent.pth 3 4 5 5 4 11 12 13 5 6 8 3 1 4`
	Repeat the command until you get a satisfactory result, because it generates a different molecule everytime

5. To generate multiple molecules but output only the best molecules for a custom path
	`python3 -B RL_6.py -b agent.pth Cx Cy Cz a b o j w T1x T1y T1z T2x T2y T2z ...`
	example:
	`python3 -B RL_6.py -b agent.pth 3 4 5 5 4 11 12 13 5 6 8 3 1 4`
	The agent will perform 300 attempts

 for a custom path and targets:

The following are SLURM and PBS job submission scripts to train the RL agent on a high performance supercomputer:
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

import os
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
parser.add_argument('-t', '--train', action='store_true', help='Train a reinforcement learning agent')
parser.add_argument('-p', '--play', nargs='+', help='Have a trained agent play the game using the agent.pth file')
parser.add_argument('-g', '--generate', nargs='+', help='Have a trained agent generate a molecule for a custom path')
parser.add_argument('-b', '--batch', nargs='+', help='Have a trained agent generate the best molecules for a custom path')
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
	timesteps     = 10e7
	n_minibatches = 128
	epochs        = 16
	seed          = 1
	lr            = 2.5e-4
	gamma         = 0.95
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
	print('Training on:', device, '||', 'Started on:', date, '\n' + '='*54)
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
		# Steps to generate a dataset
		Gts, Lns = [], []
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
					Ln = round(e['episode']['l'], 3)
					Gts.append(Gt)
					Lns.append(Ln)
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
		time_minutes = round(time_seconds/60, 1)
		if log:
			with open('train.log', 'a') as f:
				Gt_mean = round(np.array(Gts).mean(), 3)
				Gt_SD   = round(np.array(Gts).std(), 3)
				Ln_mean = round(np.array(Lns).mean(), 3)
				Ln_SD   = round(np.array(Lns).std(), 3)
				A_loss  = round(pg_loss.item(), 3)
				C_loss  = round(v_loss.item(), 3)
				Entropy = round(entropy_loss.item(), 3)
				Loss    = round(loss.item(), 3)
				KL      = round(approx_kl.item(), 3)
				Clip    = round(clipfracs[-1], 3)
				exp_var = round(explained_var, 3)
				A = f'Update: {update:>5,} / {n_updates:<10,}'
				B = f'Steps: {global_step:<15,}'
				C = f'Returns: {Gt_mean:<9,} +- {Gt_SD:<10,}'
				D = f'Lengths: {Ln_mean:<6,} +- {Ln_SD:<10,}'
				E = f'A_loss: {A_loss:<10,}'
				F = f'C_loss: {C_loss:<15,}'
				G = f'Entropy loss: {Entropy:<10,}'
				H = f'Final loss: {Loss:<15,}'
				I = f'KL: {KL:<10,}'
				J = f'Clip: {Clip:<10,}'
				K = f'Explained Variance: {exp_var:<10,}'
				L = f'Minutes per update: {time_minutes:<6,}'
				M = f'Remaining time: {time_update}'
				f.write(A + B + C + D + E + F + G + H + I + J + K + L + M + '\n')
			# Export agent model every 50 updates
			if (update % 50 == 0): 
				# Export agent model
				torch.save(agent, f'agent_{update}.pth')
		print(f'{B}{C}')
	# Export agent model
	torch.save(agent, 'agent.pth')

def play(filename='agent.pth', custom=[]):
	''' Play the MiraMar environment using a trained PPO agent '''
	# Import agent model
	agent = torch.load(filename)
	agent.eval()
	# Play game
	env = MiraMar()
	if custom != []:
		C       = custom[0]
		a, b    = custom[1], custom[2]
		o, j, w = custom[3], custom[4], custom[5]
		targets = custom[6]
		S, I = env.reset(custom=[C, a, b, o, j, w, targets])
	else:
		S, I = env.reset()
	done = False
	Gt = 0
	while not done:
		S = torch.Tensor([S]).to('cpu')
		A, _, _, _ = agent.get_action_and_value(S)
		S, R, T, U, I = env.step(A[0].numpy())
		Gt += R
		done = bool(T or U)
	print('Actions:', I['actions'])
	print('Rewards:', I['rewards'])
	print('Episode:', I['episode'])
	env.render()

def batch():
	'''
	Play the MiraMar environment using a trained PPO agent 300 times and only
	output the molecules with the highest reward and shortest N-term to
	C-term distance, logging only the successful attempts. In other words: play
	the environment and output only the	best cyclic peptides.
	'''
	C = [float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])]
	a = float(sys.argv[6])
	b = float(sys.argv[7])
	o = float(sys.argv[8])
	j = float(sys.argv[9])
	w = float(sys.argv[10])
	targets = [float(x) for x in sys.argv[11:]]
	targets = [targets[i:i+3] for i in range(0, len(targets), 3)]
	best = 0
	n = 300
	for iters in range(n):
		Range = (1.0 - (iters - 1.0) / n) * 1000
		agent = torch.load(sys.argv[2])
		agent.eval()
		env = MiraMar()
		S, I = env.reset(custom=[C, a, b, o, j, w, targets])
		done = False
		Gt = 0
		while not done:
			S = torch.Tensor([S]).to('cpu')
			A, _, _, _ = agent.get_action_and_value(S)
			S, R, T, U, I = env.step(A[0].numpy())
			Gt += R
			done = bool(T or U)
		C_term = np.linalg.norm(env.pose.GetAtom(env.i, 'C') - env.pose.GetAtom(0, 'N'))
		GT = round(Gt, 3)
		TR = round(C_term, 3)
		print(f'Attempt: {iters}\tGt = {GT}\tC_term = {TR}')
		if best-Range < Gt and 1.5 < C_term < 3.0:
			best = Gt
			with open('output.log', 'a') as f:
				ACTIONS = I['actions']
				REWARDS = I['rewards']
				EPISODE = I['episode']
				f.write(f'{iters}:\n')
				f.write(f'Actions: {ACTIONS}:\n')
				f.write(f'Rewards: {REWARDS}:\n')
				f.write(f'Episode: {EPISODE}:\n')
			env.render(show=False, save=True)
			os.rename('molecule.pdb', f'molecule_{iters}.pdb')

def main():
	if args.train:      train()
	elif args.play:
		play(filename=sys.argv[2])
	elif args.generate:
		Cx = float(sys.argv[3])
		Cy = float(sys.argv[4])
		Cz = float(sys.argv[5])
		a  = float(sys.argv[6])
		b  = float(sys.argv[7])
		o  = float(sys.argv[8])
		j  = float(sys.argv[9])
		w  = float(sys.argv[10])
		T = [float(x) for x in sys.argv[11:]]
		T = [T[i:i+3] for i in range(0, len(T), 3)]
		play(filename=sys.argv[2], custom=[[Cx, Cy, Cz], a, b, o, j, w, T])
	elif args.batch:
		batch()

if __name__ == '__main__': main()
