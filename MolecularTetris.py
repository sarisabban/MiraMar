import os
import sys
import math
import shutil
import argparse
import warnings
import subprocess
import numpy as np
from Pose.pose import *
from pathlib import Path
from gym.spaces import Dict, Discrete, Box, Tuple, MultiDiscrete
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='MolecularTetris Gam')
parser.add_argument('-p', '--play', action='store_true',
help='Manually play the game')
parser.add_argument('-rl', '--rl_train', action='store_true',
help='Train a reinforcement learning agent')
parser.add_argument('-rlp', '--rl_play', nargs='+',
help='Have an agent play the game using the policy file')
args = parser.parse_args()

class MolecularTetris():
	''' Game for designing cyclic peptides using reinforcement learning '''
	def __init__(self):
		''' Initialise global variables '''
		self.observation_space = Box(
			low=np.array( [0,  0, 0,   0, -50, 0, 0,   0, 0, -50]),
			high=np.array([1, 15, 1, 360,  50, 1, 7, 360, 7,  50]),
			dtype=np.float32)
		self.action_space = Discrete(8)
		self.n = None
	def get_angle_meanings(self, action):
		''' Definition of each action's angle '''
		angles = {0:0, 1:45, 2:90, 3:135, 4:180, 5:225, 6:270, 7:315}
		return(angles[action])
	def get_residue_meanings(self, action):
		''' Definition of each action's residue '''
		residues = {
				0:'A',   1:'C',  2:'D',  3:'E',  4:'F',
				5:'G',   6:'H',  7:'I',  8:'K',  9:'L',
				10:'M', 11:'N', 12:'P', 13:'Q', 14:'R',
				15:'S', 16:'T', 17:'V', 18:'W', 19:'Y'}
		return(residues[action])
	def seed(self, n=None):
		''' Change the game's random seed that initiates the environment '''
		self.n = n
	def RotationMatrix(self, thetaX, thetaY, thetaZ):
		''' Rotation Matrix '''
		sx = math.sin(math.radians(thetaX))
		cx = math.cos(math.radians(thetaX))
		sy = math.sin(math.radians(thetaY))
		cy = math.cos(math.radians(thetaY))
		sz = math.sin(math.radians(thetaZ))
		cz = math.cos(math.radians(thetaZ))
		Rx = np.array([[1,  0,   0], [0, cx, -sx], [0, sx,  cx]])
		Ry = np.array([[ cy, 0, sy], [  0, 1,  0], [-sy, 0, cy]])
		Rz = np.array([[cz, -sz, 0], [sz,  cz, 0], [0,    0, 1]])
		R  = Rz.dot(Ry).dot(Rx)
		return(R)
	def path(self, path=False, axis=False):
		''' Generate an elliptical path and return important metrics '''
		a, b, o, j, w = self.a, self.b, self.o, self.j, self.w
		R  = self.RotationMatrix(o, j, w)
		C  = np.array(self.C)
		e  = math.sqrt(1-(b**2/a**2))
		c  = math.sqrt(a**2 - b**2)
		y  = math.sqrt((1 - a**2/a**2) * b**2)
		F1 = [C[0]-c, C[1], C[2]]
		F2 = [C[0]+c, C[1], C[2]]
		F1 = C - F1
		F2 = C - F2
		F1 = np.matmul(F1, R)
		F2 = np.matmul(F2, R)
		F1 = C + F1
		F2 = C + F2
		start = np.array([C[0], C[1], C[2]]) - np.array([a, y, 0])
		start = C - start
		start = np.matmul(start, R)
		start_mag = np.linalg.norm(start)
		start = C + start
		vX = F1 - F2
		X  = (vX / np.linalg.norm(vX))
		y  = math.sqrt(b**2)
		y  = np.array([C[0], C[1], C[2]]) - np.array([0, y, 0])
		y  = C - y
		vY = -np.matmul(y, R)
		Y  = (vY / np.linalg.norm(vY))
		vZ = np.cross(vY, vX)
		Z  = (vZ / np.linalg.norm(vZ))
		if path:
			points = []
			points.append(C)
			points.append(F1)
			points.append(F2)
			for x in np.arange(-a, a, 1):
				y2 = (1 - x**2/a**2) * b**2
				yt =  math.sqrt(y2)
				yb = -math.sqrt(y2)
				ut = np.array(C) - np.array([x, yt, 0])
				ub = np.array(C) - np.array([x, yb, 0])
				ut = C - ut
				ub = C - ub
				ut = np.matmul(ut, R)
				ub = np.matmul(ub, R)
				ut = C + ut
				ub = C + ub
				points.append(ut)
				points.append(ub)
			return(points)
		if axis: return(start, F1, F2, e, X, Y, Z)
		else:    return(start, F1, F2, e)
	def project(self, atom):
		''' Project an atom onto the ellipse to get point P and its angle T '''
		C, a, b = self.C, self.a, self.b
		R  = self.RotationMatrix(self.o, self.j, self.w)
		R_ = np.linalg.inv(R)
		atom_ = atom - C
		atom_ = np.matmul(atom_, R_)
		T  = math.atan2(atom_[1], atom_[0])
		r  = (a*b)/math.sqrt(a**2 * math.sin(T)**2 + b**2 * math.cos(T)**2)
		P_ = np.array([r * math.cos(T), r * math.sin(T), 0])
		P  = np.matmul(P_, R)
		P  = C + P
		vP = P - atom
		vP_mag = np.linalg.norm(vP)
		T  = T * 180 / math.pi
		if T < 0: T += 360.0
		if T == 0.0: T = 360.0
		return(T, P, vP, vP_mag, r)
	def render(self, show=False, save=True, filename='molecule'):
		''' Export the molecule as a PDB file and display it '''
		points = self.path(path=True)
		with open('path.pdb', 'w') as F:
			for i, p in enumerate(points):
				if    i == 0: a, e, c = 'N', 'N', 'A' ; F.write('HEADER C\n')
				elif  i == 1: a, e, c = 'O', 'O', 'A' ; F.write('HEADER F1\n')
				elif  i == 2: a, e, c = 'O', 'O', 'A' ; F.write('HEADER F2\n')
				elif  i == 3: a, e, c = 'H', 'H', 'B' ; F.write('HEADER Path\n')
				else: a, e, c = 'H', 'H', 'B'
				A, l, r, s, I = 'ATOM', '', 'GLY', 1, ''
				x, y, z, o, t, q = p[0], p[1], p[2], 1.0, 1.0, 0.0
				Entry = self.pose.PDB_entry(A,i+1,a,l,r,c,s,I,x,y,z,o,t,q,e)
				F.write(Entry)
		self.pose.Export('{}.pdb'.format(filename))
		if Path('points.pdb').exists():
			display = ['pymol', 'molecule.pdb', 'path.pdb', 'points.pdb']
			remove = ['rm', 'molecule.pdb', 'path.pdb', 'points.pdb']
		else:
			display = ['pymol', 'molecule.pdb', 'path.pdb']
			remove = ['rm', 'molecule.pdb', 'path.pdb']
		locate = shutil.which('pymol')
		if show == True and save == True:
			if locate == None:
				print('PyMOL not installed')
				return
			subprocess.run(display, capture_output=True)
		elif show == True and save == False:
			if locate == None:
				print('PyMOL not installed')
				return
			subprocess.run(display, capture_output=True)
			subprocess.run(remove)
		elif show == False and save == True:
			return
		elif show == False and save == False:
			subprocess.run(remove)
	def export(self, P, atom, define):
		''' Export specific point '''
		with open('points.pdb', 'a') as F:
			F.write(f'HEADER {define}\n')
			a, e, c = atom, atom, 'C'
			A, l, r, s, I = 'ATOM', '', 'GLY', 1, ''
			x, y, z, o, t, q = P[0], P[1], P[2], 1.0, 1.0, 0.0
			Entry = self.pose.PDB_entry(A,1,a,l,r,c,s,I,x,y,z,o,t,q,e)
			F.write(Entry)
	def position(self):
		''' Position the polypeptide at the start position of the path '''
		CA = self.pose.GetAtom(0, 'CA')
		AAcoord = self.pose.data['Coordinates']
		AAcoord = (self.start - CA) + AAcoord
		Ax, Ay, Az = 180+self.o, 0-self.j, -45-self.w
		R = self.RotationMatrix(Ax, Ay, Az)
		AAcoord = AAcoord - self.start
		AAcoord = np.matmul(AAcoord, R)
		AAcoord = AAcoord + self.start
		self.pose.data['Coordinates'] = AAcoord
	def addAA(self, AA='G', phi=None, psi=None):
		''' Add one amino acid to the end of the polypeptide chain '''
		if self.pose == None:
			self.sequence = []
			self.PHIs = [0.0]
			self.PSIs = [180.0]
			self.pose = Pose()
			self.pose.Build(AA)
			self.position()
			self.sequence.append(AA)
		else:
			if psi == None: psi = 180.0
			if phi == None: phi = 180.0
			self.sequence.append(AA)
			self.PHIs.append(phi)
			self.PSIs.append(psi)
			self.pose = Pose()
			self.pose.Build(''.join(self.sequence))
			self.position()
			for i, (p, s) in enumerate(zip(self.PHIs, self.PSIs)):
				self.pose.Rotate(i, p, 'PHI')
				self.pose.Rotate(i, s, 'PSI')
	def reset(self):
		''' Reset game '''
		self.pose = None
		np.random.seed(self.n)
		self.i = 0
		self.C = np.random.uniform(0, 50, size=(3,))
		self.b = np.random.uniform(2, 5)
		self.a = np.random.uniform(self.b, 10)
		self.o = np.random.uniform(0, 90)
		self.j = np.random.uniform(0, 90)
		self.w = np.random.uniform(0, 90)
		self.start, F1, F2, e = self.path()
		self.addAA()
		self.T = 360
		self.F1P = 0
		self.switch = 0
		S, R, St, info, data = self.SnR(self.start, F1, F2, e)
		return(S, data)
	def step(self, action):
		''' Play one step, add amino acid and define its phi/psi angles '''
		AA  = 'G'
		phi = self.get_angle_meanings(action)
		psi = None
		self.addAA(AA, phi, psi)
		self.i = max(self.pose.data['Amino Acids'].keys())
		start, F1, F2, e = self.path()
		return(self.SnR(start, F1, F2, e))
	def AminoAcidOri(self):
		''' Get amino acid origin and its axis '''
		N  = self.pose.GetAtom(self.i, 'N')
		CA = self.pose.GetAtom(self.i, 'CA')
		C  = self.pose.GetAtom(self.i, 'C')
		vNCA = N - CA
		vCAC = CA - C
		vNCA_mag = np.linalg.norm(vNCA)
		vNCA = vNCA*(3.716521828269005/vNCA_mag)
		vNCA_mag = np.linalg.norm(vNCA)
		origin = N - vNCA
		vX = CA - N
		X = vX / np.linalg.norm(vX)
		X_mag = np.linalg.norm(X)
		vZ = np.cross(vX, vCAC)
		Z = vZ / np.linalg.norm(vZ)
		Z_mag = np.linalg.norm(Z)
		vY = np.cross(vX, vZ)
		Y = vY / np.linalg.norm(vY)
		Y_mag = np.linalg.norm(Y)
		return(origin, X, Y, Z)
	def future(self, phi=0, F1=[0, 0, 0], F2=[0, 0, 0], plot=False):
		''' For a phi angle return the distance of fCA and the angle of P '''
		phi, PHI = math.radians(phi), phi
		a, b = self.a, self.b
		oriA, XA, YA, ZA = self.AminoAcidOri()
		d = 3.1870621267869894
		fCA = oriA + YA * d
		T, fP, _, _, radius = self.project(fCA)
		vCAP = fP - fCA
		vCAP_mag = np.linalg.norm(fP - fCA)
		R  = self.RotationMatrix(self.o, self.j, self.w)
		R_ = np.linalg.inv(R)
		RM = np.array([
		[XA[0], XA[1], XA[2]],
		[YA[0], YA[1], YA[2]],
		[ZA[0], ZA[1], ZA[2]]])
		RM_ = np.linalg.inv(RM)
		fCA_phi = [0, d * math.cos(phi), d * math.sin(phi)]
		C_ori = np.matmul(self.C - oriA, RM_)
		F1_ori = np.matmul(F1 - oriA, RM_)
		F2_ori = np.matmul(F2 - oriA, RM_)
		fCA_x = np.matmul(fCA_phi, RM)
		fCA_x = fCA_x + oriA
		fCA_x = fCA_x - self.C
		fCA_x = np.matmul(fCA_x, R_)
		theta = math.atan2(fCA_x[1], fCA_x[0])
		r = (a*b)/math.sqrt(a**2*math.sin(theta)**2 + b**2*math.cos(theta)**2)
		fP_phi = np.array([r * math.cos(theta), r * math.sin(theta), 0])
		fP_phi = np.matmul(fP_phi, R)
		fP_phi = fP_phi + self.C
		fP_phi = np.matmul(fP_phi - oriA, RM_)
		vCAP_phi = fP_phi - fCA_phi
		vCAP_phi_mag = np.linalg.norm(vCAP_phi)
		T = theta * 180 / math.pi
		if T < 0: T += 360.0
		if T == 0.0: T = 360.0
		if plot:
			self.export(oriA, 'N', 'Amino Acid Origin')
			XA = [XA[0]+oriA[0], XA[1]+oriA[1], XA[2]+oriA[2]]
			YA = [YA[0]+oriA[0], YA[1]+oriA[1], YA[2]+oriA[2]]
			ZA = [ZA[0]+oriA[0], ZA[1]+oriA[1], ZA[2]+oriA[2]]
			self.export(XA, 'I', 'Amino Acid X-axis')
			self.export(YA, 'I', 'Amino Acid Y-axis')
			self.export(ZA, 'I', 'Amino Acid Z-axis')
			X, Y, Z = [1, 0, 0], [0, 1, 0], [0, 0, 1]
			self.export([0, 0, 0], 'C', 'Global Origin')
			self.export(X, 'H', 'Global X-axis')
			self.export(Y, 'H', 'Global Y-axis')
			self.export(Z, 'H', 'Global Z-axis')
			self.export(C_ori, 'S', 'C_ori')
			self.export(F1_ori, 'O', 'F1_ori')
			self.export(F2_ori, 'O', 'F2_ori')
			self.export(fCA, 'S', 'fCA')
			self.export(fP, 'I', 'fP')
			self.export(fCA_phi, 'S', f'fCA_phi_{PHI}')
			self.export(fP_phi, 'I', f'fP_phi_{PHI}')
			R = self.RotationMatrix(self.o, self.j, self.w)
			for x in np.arange(-a, a, 1):
				y2 = (1 - x**2/a**2) * b**2
				yt =  math.sqrt(y2)
				yb = -math.sqrt(y2)
				ut = np.array(self.C) - np.array([x, yt, 0])
				ub = np.array(self.C) - np.array([x, yb, 0])
				ut = self.C - ut
				ub = self.C - ub
				ut = np.matmul(ut, R)
				ub = np.matmul(ub, R)
				ut = self.C + ut
				ub = self.C + ub
				pt = np.matmul(ut - oriA, RM_)
				pb = np.matmul(ub - oriA, RM_)
				self.export(pt, 'H', f't{x}')
				self.export(pb, 'H', f'b{x}')
		return(T, vCAP_phi_mag)
	def SnR(self, start, F1, F2, e):
		''' Return the state features and rewards after each game step '''
		# Calculating future CA
		oriA, XA, YA, ZA = self.AminoAcidOri()
		d = 3.1870621267869894
		fCA = oriA + YA * d
		# Projected angle and distance of current CA atom
		CA = self.pose.GetAtom(self.i, 'CA')
		T, P, _, d, radius = self.project(CA)
		###########################
		##### Reward Function #####
		###########################
		R = 0
		# Reward for moving forward
		if self.T > T: R += 1
		else:          R -= 1
		self.T = T
		# Penalty for distance from ellipse surface
#		R -= d
		# F1->CA & F2->CA distance
		F1CA = np.linalg.norm(F1 - CA)
		F2CA = np.linalg.norm(F2 - CA)
		# F1->P & F2->P distance
		F1P = np.linalg.norm(F1 - P)
		F2P = np.linalg.norm(F2 - P)
		# Reward for being outside ellipse
		if F1CA > F1P and F2CA > F2P: R += 1
		else:                         R -= 1
		# Reward for going around ellipse clockwise
		if T < 180: self.switch = 1
		if   self.switch == 0 and self.F1P < F1P: R += 1
		elif self.switch == 1 and self.F1P > F1P: R += 1
		else:                                     R -= 1
		self.F1P = F1P
		###########################
		######## Features #########
		###########################
		# Check if step is odd or even
		if   (self.i % 2) != 0: OE = 0
		elif (self.i % 2) == 0: OE = 1
		# Determine which action leads to lowest fT and which to lowest fd
		Ts, ds = {}, {}
		for action in range(self.action_space.n):
			phi = self.get_angle_meanings(action)
			fT, fd = self.future(phi=phi, F1=F1, F2=F2)
			Ts[action] = fT
			ds[action] = fd
		min_fT_a = [key for key in Ts if Ts[key] == min(Ts.values())][0]
		min_fd_a = [key for key in Ts if ds[key] == min(ds.values())][0]
		min_fT_v = Ts[min_fT_a]
		min_fd_v = ds[min_fd_a]
		S = np.array([
			e, self.i, OE, T, d, self.switch,
			min_fT_a, min_fT_v, min_fd_a, min_fd_v])
		###########################
		### End State Condition ###
		###########################
		St = False
		# If polypeptide reaches 15 amino acids
		if self.i >= 15:
			St = True
		# End game if the chain made a circle onto itself
		CAs   = [self.pose.GetAtom(x, 'CA') for x in range(self.i)]
		VECs  = [CA - fCA for CA in CAs]
		MAGs  = [np.linalg.norm(VEC) for VEC in VECs]
		CHECK = [1 if x < 1.5 else 0 for x in MAGs]
		if 1 in CHECK:
			St = True
			# Reward at this end state only
			R = self.i - 15
		###########################
		####### Extra Info ########
		###########################
		info, data = None, {}
		return(S, R, St, info, data)

def play(show=True):
	''' Manually play the game '''
	print('\n' + '='*95)
	print('''\
	╔╦╗┌─┐┬  ┌─┐┌─┐┬ ┬┬  ┌─┐┬─┐  ╔╦╗┌─┐┌┬┐┬─┐┬┌─┐
	║║║│ ││  ├┤ │  │ ││  ├─┤├┬┘   ║ ├┤  │ ├┬┘│└─┐
	╩ ╩└─┘┴─┘└─┘└─┘└─┘┴─┘┴ ┴┴└─   ╩ └─┘ ┴ ┴└─┴└─┘
	           0                            +
	                                       +   C
	   7          /    1         O-H       +  /|
	             /               |        +  / |
	            /              O=C        + /  |
	6          0          2       \       +/   |
	                             H-Ca-----P    |
	        ACTIONS               /        +   |
	   5               3       H-N         +   |
	                             |          +  F1
	           4                              +\n''')
	print('='*95)
	seed = input('\nChoose a seed value, empty for a random seed, q to quit > ')
	if    seed == '': seed = None
	elif  seed == 'q': exit()
	else: seed = int(seed)
	print('\n' + '-'*95)
	print(' '*25, 'Features', ' '*50, 'Reward')
	print('-'*85 + '|' + '-'*9)
	env = MolecularTetris()
	env.seed(seed)
	obs = env.reset()
	n = env.observation_space.shape[0] - 1
	types = ['e', 'i', 'OE', 'T', 'mag', 'Switch', 'Ta', 'Tv', 'Da', 'Dv']
	title = ''
	for F in types: title += '{:<{}}'.format(F, n)
	print(title)
	output = ''
	for F in obs[0]: F = round(F, 1) ; output += '{:<{}}'.format(F, n)
	print(output)
	Gt = []
	St = False
	As = [x for x in range(env.action_space.n)]
	actions = []
	while (St == False):
		inp = input('Action [0, 7] q to quit > ')
		if inp == 'q': exit()
		try: inp = int(inp)
		except: print('incorrect input') ; continue
		if inp not in As: print('incorrect input') ; continue
		actions.append(inp)
		obs = env.step(inp)
		R = round(obs[1], 3)
		output = ''
		for F in obs[0]: F = round(F, 1) ; output += '{:<{}}'.format(F, n)
		output += '  {:<5}'.format(R)
		print(output)
		Gt.append(R)
		St = obs[2]
	print('='*95)
	print('Actions:', actions)
	print('-'*20)
	print('Total Reward =', sum(Gt))
	if show: env.render(show=True, save=False)

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

def RL(epochs=1, play=False, filename='policy.pth'):
	''' Reinforcement Learning setup '''
	import torch
	import tianshou
	from torch import nn
	from tianshou.policy import PPOPolicy, DQNPolicy
	from tianshou.utils.net.discrete import Actor, Critic
	from tianshou.utils.net.common import ActorCritic, Net
	from tianshou.env import DummyVectorEnv, SubprocVectorEnv
	from tianshou.data import Collector, VectorReplayBuffer, Batch
	from tianshou.trainer import onpolicy_trainer, offpolicy_trainer
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print(device)
	# 0. Get information about the environment
	env = MolecularTetris()
	features = env.observation_space.shape
	actions  = env.action_space.n
	# 1. Setup the training and testing environments
	train = SubprocVectorEnv([lambda:env for _ in range(200)])
	tests = SubprocVectorEnv([lambda:env for _ in range(150)])
	# 2. Setup neural networks and policy - PPO
	net = Net(features, hidden_sizes=[64, 64, 64], device=device)
	actor = Actor(net, actions, device=device).to(device)
	critic = Critic(net, device=device).to(device)
	actor_critic = ActorCritic(actor, critic)
	optim = torch.optim.Adam(actor_critic.parameters(), lr=1e-4)
	dist = torch.distributions.Categorical
	policy = PPOPolicy( \
	actor, critic, optim, dist, action_space=env.action_space, \
	deterministic_eval=True)
	# 3. Setup vectorised replay buffer
	VRB = VectorReplayBuffer(20000, len(train))
	# 4. Setup collectors
	train_collector = Collector(policy, train, VRB)
	tests_collector = Collector(policy, tests)
	if not play:
		# 5. Train
		result = onpolicy_trainer(
			policy,
			train_collector,
			tests_collector,
			step_per_epoch=50000,
			max_epoch=epochs,
			step_per_collect=20,
			episode_per_test=10,
			repeat_per_collect=10,
			batch_size=256,
			stop_fn=lambda mean_reward: mean_reward >= 50)
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
	if   args.play:     play()
	elif args.rl_train: RL(epochs=100)
	elif args.rl_play:  RL(epochs=0, play=True, filename=sys.argv[2])

if __name__ == '__main__': main()
