# 1. Install : pip install torch numpy scipy gymnasium git+https://github.com/sarisabban/Pose
# 2. Training: python3 -B RL_PPO.py
# 3. Play    : python3 -B RL_PPO.py
# This algorithm is a derivation from CleanRL's Multidescrete mask PPO script
# https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

import time
import torch
import random
import datetime
import numpy as np
import gymnasium as gym
from MolecularTetris import MolecularTetris

env             = MolecularTetris()
seed            = 1
num_envs        = 32
num_steps       = 32
total_timesteps = 2000000
num_minibatches = 4
update_epochs   = 4
learning_rate   = 2.5e-4
gamma           = 0.99
gae_lambda      = 0.95
clip_coef       = 0.1
ent_coef        = 0.01
vf_coef         = 0.5
max_grad_norm   = 0.5
mask_actions    = False
target_kl       = None
log             = True

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
	torch.nn.init.orthogonal_(layer.weight, std)
	torch.nn.init.constant_(layer.bias, bias_const)
	return layer

class CategoricalMasked(torch.distributions.categorical.Categorical):
	def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
		self.masks = masks
		if len(self.masks) == 0:
			super(CategoricalMasked,self).__init__(probs, logits, validate_args)
		else:
			self.masks = masks.type(torch.BoolTensor).to(device)
			logits=torch.where(self.masks,logits,torch.tensor(-1e8).to(device))
			super(CategoricalMasked,self).__init__(probs, logits, validate_args)
	def entropy(self):
		if len(self.masks) == 0:
			return super(CategoricalMasked, self).entropy()
		p_log_p = self.logits * self.probs
		p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to(device))
		return -p_log_p.sum(-1)

class Agent(torch.nn.Module):
	def __init__(self, envs):
		super(Agent, self).__init__()
		obs_shape = envs.single_observation_space.shape
		self.network = torch.nn.Sequential(
			layer_init(torch.nn.Linear(np.array(obs_shape).prod(), 64)),
			torch.nn.ReLU(),
			layer_init(torch.nn.Linear(64, 64)),
			torch.nn.ReLU(),
			torch.nn.Flatten(),
			layer_init(torch.nn.Linear(64, 128)),
			torch.nn.ReLU(),)
		self.nvec = envs.single_action_space.nvec
		self.actor = layer_init(torch.nn.Linear(128, self.nvec.sum()), std=0.01)
		self.critic = layer_init(torch.nn.Linear(128, 1), std=1)
	def get_value(self, x):
		return self.critic(self.network(x))
	def get_action_and_value(self, x, action_mask, action=None):
		hidden = self.network(x)
		logits = self.actor(hidden)
		split_logits = torch.split(logits, self.nvec.tolist(), dim=1)
		split_action_masks = torch.split(action_mask, self.nvec.tolist(), dim=1)
		multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_action_masks)]
		if action is None:
			action = torch.stack([categorical.sample() for categorical in multi_categoricals])
		logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
		entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
		return action.T, logprob.sum(0), entropy.sum(0), self.critic(hidden)

# Fix seed to make all experiments reproducible
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Training on:', device)
# Environment setup
envs = gym.vector.SyncVectorEnv([lambda: env for i in range(num_envs)])
agent = Agent(envs).to(device)
optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
batch_size = int(num_envs * num_steps)
minibatch_size = int(batch_size // num_minibatches)
num_updates = total_timesteps // batch_size
# Storage setup
s_obs_space = envs.single_observation_space.shape
s_act_space = envs.single_action_space.shape
s_acts_nvec = envs.single_action_space.nvec.sum()
obs = torch.zeros((num_steps, num_envs) + s_obs_space).to(device)
actions = torch.zeros((num_steps, num_envs) + s_act_space).to(device)
rewards = torch.zeros((num_steps, num_envs)).to(device)
values = torch.zeros((num_steps, num_envs)).to(device)
dones = torch.zeros((num_steps, num_envs)).to(device)
logprobs = torch.zeros((num_steps, num_envs)).to(device)
action_masks = torch.zeros((num_steps, num_envs) + (s_acts_nvec,)).to(device)
# Start the environment
next_obs = torch.Tensor(envs.reset(seed=seed)[0]).to(device)
next_done = torch.zeros(num_envs).to(device)
# Updates
global_step = 0
for update in range(1, num_updates + 1):
	time_start = time.time()
	# Anneal the learning rate
	frac = 1.0 - (update - 1.0) / num_updates
	lrnow = frac * learning_rate
	optimizer.param_groups[0]['lr'] = lrnow
	# Steps
	Gts = []
	for step in range(0, num_steps):
		global_step += 1 * num_envs
		obs[step] = next_obs
		dones[step] = next_done
		if mask_actions:
			action_masks[step] = \
			torch.Tensor(np.array([env.action_mask for env in envs.envs]))
		else:
			sum_actions = envs.single_action_space.nvec.sum()
			masks = [0.0 for a in range(sum_actions)]
			array = np.array([masks for env in envs.envs])
			action_masks[step] = torch.Tensor(array)
		# Play game
		with torch.no_grad():
			action, logprob, _, value = \
			agent.get_action_and_value(next_obs, action_masks[step])
			values[step] = value.flatten()
		actions[step] = action
		logprobs[step] = logprob
		################################################
		next_obs, reward, term, trun, info = envs.step(action.cpu().numpy())
		done = term + trun
		index = np.where(done==True)[0]
		rewards[step] = torch.tensor(reward).to(device).view(-1)
		next_obs = torch.Tensor(next_obs).to(device)
		next_done = torch.Tensor(done).to(device)
		if 'final_info' in info.keys():
			for e in info['final_info'][index]:
				Gt = round(e['episode']['r'], 3)
				Gts.append(Gt)
		###############################################
	Gt_mean = round(np.array(Gts).mean(), 3)
	Gt_SD   = round(np.array(Gts).std(), 3)
	# Bootstrap value
	with torch.no_grad():
		next_value = agent.get_value(next_obs).reshape(1, -1)
		advantages = torch.zeros_like(rewards).to(device)
		lastgaelam = 0
		for t in reversed(range(num_steps)):
			if t == num_steps - 1:
				nextnonterminal = 1.0 - next_done
				nextvalues = next_value
			else:
				nextnonterminal = 1.0 - dones[t + 1]
				nextvalues = values[t + 1]
			delta = \
			rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
			advantages[t] = lastgaelam = \
			delta + gamma * gae_lambda * nextnonterminal * lastgaelam
		returns = advantages + values
	# Flatten the batch
	b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
	b_logprobs = logprobs.reshape(-1)
	b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
	b_advantages = advantages.reshape(-1)
	b_returns = returns.reshape(-1)
	b_values = values.reshape(-1)
	b_action_masks = action_masks.reshape((-1, action_masks.shape[-1]))
	# Optimise the policy (Actor) and value (Critic) neural networks
	b_inds = np.arange(batch_size)
	clipfracs = []
	for epoch in range(update_epochs):
		np.random.shuffle(b_inds)
		for start in range(0, batch_size, minibatch_size):
			end = start + minibatch_size
			mb_inds = b_inds[start:end]
			_, newlogprob, entropy, newvalue = agent.get_action_and_value(
				b_obs[mb_inds],
				b_action_masks[mb_inds],
				b_actions.long()[mb_inds].T)
			logratio = newlogprob - b_logprobs[mb_inds]
			ratio = logratio.exp()
			# Aproximate KL divergence
			with torch.no_grad():
				old_approx_kl = (-logratio).mean()
				approx_kl = ((ratio - 1) - logratio).mean()
				clipfracs += [((ratio - 1.0).abs() > clip_coef) \
				.float().mean().item()]
			mb_advantages = b_advantages[mb_inds]
			mb_advantages = \
			(mb_advantages - mb_advantages.mean()) / (mb_advantages.std()+1e-8)
			# Policy loss
			pg_loss1 = -mb_advantages*ratio
			pg_loss2 = -mb_advantages*torch.clamp(ratio,1-clip_coef,1+clip_coef)
			pg_loss = torch.max(pg_loss1, pg_loss2).mean()
			# Value loss
			newvalue = newvalue.view(-1)
			v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
			v_clipped = b_values[mb_inds] + torch.clamp(
				newvalue - b_values[mb_inds], -clip_coef, clip_coef)
			v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
			v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
			v_loss = 0.5 * v_loss_max.mean()
			# Entropy loss
			entropy_loss = entropy.mean()
			# Final loss
			loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
			# Backpropagation
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
			optimizer.step()
		if target_kl is not None:
			if approx_kl > target_kl: break
	# Gradient descent
	y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
	var_y = np.var(y_true)
	explained_var = np.nan if var_y == 0 else 1-np.var(y_true - y_pred) / var_y
	time_seconds = time.time() - time_start
	time_update_seconds = round(time_seconds * (num_updates - update), 0)
	time_update = datetime.timedelta(seconds=time_update_seconds)
	if log:
		with open('train.log', 'a') as f:
			A_loss  = round(pg_loss.item(), 3)
			C_loss  = round(v_loss.item(), 3)
			Entropy = round(entropy_loss.item(), 3)
			Loss    = round(loss.item(), 3)
			KL      = round(approx_kl.item(), 3)
			Clip    = round(clipfracs[-1], 3)
			A = f'Update: {update:<20,}'
			B = f'Steps: {global_step:<20,}'
			C = f'Returns: {Gt_mean:,} +- {Gt_SD:<20,}'
			D = f'A_loss: {A_loss:<20,}'
			E = f'C_loss: {C_loss:<20,}'
			F = f'Entropy loss: {Entropy:<20,}'
			G = f'Final loss: {Loss:<20,}'
			H = f'KL: {KL:<20,}'
			I = f'Clip: {Clip:<20,}'
			J = f'Remaining time: {time_update}\n'
			f.write(A + B + C + D + E + F + G + H + I + J)
	print(f'Updates: {update}/{num_updates} | Steps: {global_step:<10,} Return: {Gt_mean:,} +- {Gt_SD:<10,} Remaining time: {time_update}')
