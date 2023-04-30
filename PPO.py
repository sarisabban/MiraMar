# 1. Install dependencies: pip install tqdm torch numpy scipy gymnasium git+https://github.com/sarisabban/Pose
# 2. Execute for training (training time 6 days): python3 -B PPO.py
# 3. Execute to play invironment: python3 -B PPO.py

# This algorithm is a derivation from CleanRL's Multidescrete PPO script
# 
# This is BASH code to train the environment on a SLURM-based supercomputer
'''
#!/bin/sh
#SBATCH --job-name=MolTet
#SBATCH --partition=compsci
#SBATCH --time=144:00:00
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48

cd $SLURM_SUBMIT_DIR

python3 -B PPO.py
'''

import tqdm
import torch
import numpy as np
import gymnasium as gym
from MolecularTetris import MolecularTetris

ENV           = MolecularTetris()
n_envs        = 3
n_steps       = 20
epochs        = 4
seed          = 1
lr            = 2.5e-4
timesteps     = 2000000
n_minibatches = 4
gamma         = 0.99
gae_lambda    = 0.95
clip_coef     = 0.1
vf_coef       = 0.5
ent_coef      = 0.01
max_grad_norm = 0.5
target_kl     = None

def make_env(seed, idx):
	def thunk():
		env = ENV
		env.reset(seed=seed)
		return env
	return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
	torch.nn.init.orthogonal_(layer.weight, std)
	torch.nn.init.constant_(layer.bias, bias_const)
	return layer

class Agent(torch.nn.Module):
	def __init__(self, envs):
		super(Agent, self).__init__()
		self.network = torch.nn.Sequential(
			layer_init(torch.nn.Linear(19, 128)),
			torch.nn.ReLU(),
			layer_init(torch.nn.Linear(128, 128)),
			torch.nn.ReLU(),
			layer_init(torch.nn.Linear(128, 128)),
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

# Fix seed to make all experiments reproducible
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Environment setup
envs = gym.vector.SyncVectorEnv([make_env(seed + i, i) for i in range(n_envs)])
agent = Agent(envs).to(device)
optimizer = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
batch_size = int(n_envs * n_steps)
minibatch_size = int(batch_size // n_minibatches)
n_updates = timesteps // batch_size
# Storage setup
s_obs_space = envs.single_observation_space.shape
s_act_space = envs.single_action_space.shape
obs       = torch.zeros((n_steps, n_envs) + s_obs_space).to(device)
actions   = torch.zeros((n_steps, n_envs) + s_act_space).to(device)
rewards   = torch.zeros((n_steps, n_envs)).to(device)
values    = torch.zeros((n_steps, n_envs)).to(device)
dones     = torch.zeros((n_steps, n_envs)).to(device)
logprobs  = torch.zeros((n_steps, n_envs)).to(device)
# Start the environment
next_obs  = torch.Tensor(envs.reset(seed=seed)[0]).to(device)
next_done = torch.zeros(n_envs).to(device)
# Updates
for update in range(1, n_updates + 1):
	# Anneal the learning rate
	optimizer.param_groups[0]['lr'] = (1.0 - (update - 1.0) / n_updates) * lr
	# Steps
	text = f'Update {update}/{n_updates} - Steps'
	for step in tqdm.tqdm(range(n_steps), desc=text, ascii=True, leave=False):
		obs[step] = next_obs
		dones[step] = next_done
		# Action logic
		with torch.no_grad():
			action, logprob, _, value = agent.get_action_and_value(next_obs)
			values[step] = value.flatten()
		actions[step] = action
		logprobs[step] = logprob
		# Play game
		next_obs, reward, done, _, info = envs.step(action.cpu().numpy())
		rewards[step] = torch.tensor(reward).to(device).view(-1)
		next_obs  = torch.Tensor(next_obs).to(device)
		next_done = torch.Tensor(done).to(device)
	# Bootstrap value
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
			advantages[t] = lastgaelam = \
			delta + gamma * gae_lambda * nextnonterminal * lastgaelam
		returns = advantages + values
	# Flatten the batch
	b_obs        = obs.reshape((-1,) + envs.single_observation_space.shape)
	b_actions    = actions.reshape((-1,) + envs.single_action_space.shape)
	b_returns    = returns.reshape(-1)
	b_advantages = advantages.reshape(-1)
	b_values     = values.reshape(-1)
	b_logprobs   = logprobs.reshape(-1)
	b_indx       = np.arange(batch_size)
	# Optimise the policy (Actor) and value (Critic) neural networks
	for epoch in range(epochs):
		np.random.shuffle(b_indx)
		for start in range(0, batch_size, minibatch_size):
			end = start + minibatch_size
			mb_inds = b_indx[start:end]
			_, newlogprob, entropy, newvalue = agent.get_action_and_value( \
			b_obs[mb_inds], b_actions.long()[mb_inds].T)
			logratio = newlogprob - b_logprobs[mb_inds]
			ratio = logratio.exp()
			# Aproximate KL divergence
			with torch.no_grad():
				old_approx_kl = (-logratio).mean()
				approx_kl = ((ratio - 1) - logratio).mean()
			mb_adv = b_advantages[mb_inds]
			mb_advantages = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
			# Policy loss
			pg_loss1 = -mb_advantages*ratio
			pg_loss2 = -mb_advantages*torch.clamp(ratio,1-clip_coef,1+clip_coef)
			pg_loss = torch.max(pg_loss1, pg_loss2).mean()
			# Value loss
			newvalue = newvalue.view(-1)
			v_loss_unclipped = (newvalue - b_returns[mb_inds])**2
			v_clipped = b_values[mb_inds] + torch.clamp(
				newvalue - b_values[mb_inds], clip_coef, clip_coef,)
			v_loss_clipped = (v_clipped - b_returns[mb_inds])**2
			v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
			v_loss = 0.5 * v_loss_max.mean()
			# Entropy loss
			entropy_loss = entropy.mean()
			# Final loss
			loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
			# Backpropagation
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(agent.parameters(),max_grad_norm)
			optimizer.step()
		if target_kl is not None:
			if approx_kl > target_kl: break
	# Gradient descent
	y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
	var_y = np.var(y_true)
	explained_var = np.nan if var_y == 0 else 1-np.var(y_true - y_pred) / var_y
	Gt_mean = round(rewards[-1].mean().item(), 3)
	Gt_SD   = round(rewards[-1].std().item() , 3)
	print(f'{update}\tAverage return: {Gt_mean} +- {Gt_SD}')
