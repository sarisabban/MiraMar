# 1. Install dependencies: pip install torch tianshou numpy scipy gym git+https://github.com/sarisabban/Pose
# 2. Execute for training (training time 6 days): python3 -B RL_DQN.py -rl
# 3. Execute to play invironment: python3 -B RL_DQN.py -rlp policy.pth

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

python3 -B RL_DQN.py -rl
'''

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
import torch
import argparse
from MolecularTetris import MolecularTetris
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import BranchingNet
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import DQNPolicy, BranchingDQNPolicy

parser = argparse.ArgumentParser(
description='Reinforcement learning on the MolecularTetris environment')

parser.add_argument('-rl', '--rl_train', nargs='+',
help='Train a reinforcement learning agent')

parser.add_argument('-rlp', '--rl_play', nargs='+',
help='Have a trained agent play the game using the policy.pth file')

args = parser.parse_args()

def RL_DQN(epochs=0, play=False, filename='policy.pth'):
	''' Reinforcement Learning setup '''
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print(device)
	# 0. Get information about the environment
	env = MolecularTetris()
	features = env.observation_space.shape
	actions = env.action_space.shape
	# 1. Setup the training and testing environments
	train = SubprocVectorEnv([lambda:env for _ in range(350)])
	tests = SubprocVectorEnv([lambda:env for _ in range(150)])
	# 2. Setup neural networks and policy - DQN
	net = BranchingNet(
		state_shape=features,
		num_branches=actions[0],
		common_hidden_sizes=[128, 128, 128],
		device=device).to(device)
	optim = torch.optim.Adam(net.parameters(), lr=1e-4)
	policy = BranchingDQNPolicy(
		net,
		optim,
		discount_factor=0.9,
		estimation_step=1,
		target_update_freq=320)
	# 3. Setup vectorised replay buffer
	VRB = VectorReplayBuffer(20000, len(train))
	# 4. Setup collectors
	train_collector = Collector(policy, train, VRB)
	tests_collector = Collector(policy, tests)
	if not play:
		# 5. Train
		result = offpolicy_trainer(
			policy,
			train_collector,
			tests_collector,
			step_per_epoch=2000,
			max_epoch=epochs,
			step_per_collect=20,
			episode_per_test=10,
			repeat_per_collect=10,
			batch_size=256,
			stop_fn=lambda mean_reward: mean_reward >= 25)
		print(result)
		# 6. Save policy
		torch.save(policy.state_dict(), filename)
	if play:
		# 7. Play
		tests = DummyVectorEnv([lambda:MolecularTetris() for _ in range(1)])
		tests_collector = Collector(policy, tests)
		policy.load_state_dict(torch.load(filename))
		policy.eval()
		result = tests_collector.collect(n_episode=1, render=True)
		print('Final reward: {}, length: {}' \
		.format(result['rews'].mean(), result['lens'].mean()))

def main():
	if args.rl_train:
		if sys.argv[2].upper() == 'DQN': RL_DQN(epochs=4000)
	elif args.rl_play:
		if sys.argv[2].upper() == 'DQN': RL_DQN(play=True, filename=sys.argv[3])

if __name__ == '__main__': main()