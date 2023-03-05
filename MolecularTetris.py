import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
import math
import shutil
import argparse
import warnings
import subprocess
import numpy as np
from Pose.pose import *
from pathlib import Path
from collections import defaultdict
from gym.spaces import Dict, Discrete, Box, Tuple, MultiDiscrete
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='MolecularTetris Game')
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
			low=np.array( [0,  0, 0,   0, -50, 0, 0, 0,   0, 0, 0, -50, -50]),
			high=np.array([1, 20, 1, 360,  50, 1, 7, 7, 360, 7, 7,  50,  50]),
			dtype=np.float32)
		self.action_space = MultiDiscrete([8, 8])
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
	def render(self, show=True, save=False, filename='molecule'):
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
		self.b = np.random.uniform(4, 8)
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
		phi = self.get_angle_meanings(action[0])
		psi = self.get_angle_meanings(action[1])
		self.addAA(AA, phi, psi)
		self.i = max(self.pose.data['Amino Acids'].keys())
		start, F1, F2, e = self.path()
		return(self.SnR(start, F1, F2, e))
	def AminoAcidOri(self, ori='phi'):
		''' Get amino acid origin and its axis '''
		N  = self.pose.GetAtom(self.i, 'N')
		CA = self.pose.GetAtom(self.i, 'CA')
		C  = self.pose.GetAtom(self.i, 'C')
		vNCA = N - CA
		vCAC = CA - C
		vNCA_mag = np.linalg.norm(vNCA)
		vCAC_mag = np.linalg.norm(vCAC)
		vNCA = vNCA*(3.716521828269005/vNCA_mag)
		vCAC = vCAC*(2.265083239682766/vCAC_mag)
		vNCA_mag = np.linalg.norm(vNCA)
		vCAC_mag = np.linalg.norm(vCAC)
		if ori.upper() == 'PHI':
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
		elif ori.upper() == 'PSI':
			origin = C - vCAC
			adjust = [-0.09576038, 0.1105656, 0.07848978]
			origin = origin - adjust
			vX = origin - C
			X = vX / np.linalg.norm(vX)
			X_mag = np.linalg.norm(X)
			OO = self.pose.GetAtom(self.i,'O')-self.pose.GetAtom(self.i,'OXT')
			vY = np.cross(OO, X)
			Y = vY / np.linalg.norm(vY)
			Y_mag = np.linalg.norm(Y)
			vZ = np.cross(Y, X)
			Z = vZ / np.linalg.norm(vZ)
			Z_mag = np.linalg.norm(Z)
			return(origin, X, Y, Z)
	def future(self, phi=0, psi=0, F1=[0, 0, 0], F2=[0, 0, 0], plot=False):
		''' For phi/psi angles return the distance of fCA and the angle of P '''
		a, b = self.a, self.b
		phi, psi, PHI, PSI = math.radians(phi), math.radians(psi), phi, psi
		PoriA, PXA, PYA, PZA = self.AminoAcidOri(ori='PHI')
		dp = 3.1870621267869894
		ds = 0.9526475062940741
		fCA = PoriA + PYA * dp
		PRM = np.array([
		[PXA[0], PXA[1], PXA[2]],
		[PYA[0], PYA[1], PYA[2]],
		[PZA[0], PZA[1], PZA[2]]])
		PRM_ = np.linalg.inv(PRM)
		fCA_phi = [0, dp * math.cos(phi), dp * math.sin(phi)]
		fCA_phi = np.matmul(fCA_phi, PRM) + PoriA
		SoriA, SXA, SYA, SZA = self.AminoAcidOri(ori='PSI')
		transO = SoriA - fCA
		SoriA = fCA_phi + transO
		CA = self.pose.GetAtom(self.i, 'CA')
		SXA = SoriA - CA
		SXA = SXA / np.linalg.norm(SXA)
		SZA = np.cross(SYA, SXA)
		SZA = SZA / np.linalg.norm(SZA)
		SRM = np.array([
		[SXA[0], SXA[1], SXA[2]],
		[SYA[0], SYA[1], SYA[2]],
		[SZA[0], SZA[1], SZA[2]]])
		SRM_ = np.linalg.inv(SRM)
		fCA_psi = [0, ds * math.cos(psi), ds * math.sin(psi)]
		fCA_psi = np.matmul(fCA_psi, SRM) + SoriA
		fCA = fCA_psi
		fT, fP, _, fd, fr = self.project(fCA)
		if plot:
			self.export(fCA, 'S', f'fCA_{PHI}_{PSI}')
			self.export(fP, 'I', f'fP_{PHI}_{PSI}')
		return(fT, fP, fd, fr)
	def SnR(self, start, F1, F2, e):
		''' Return the state features and rewards after each game step '''
		# Calculating future CA
		oriA, XA, YA, ZA = self.AminoAcidOri(ori='PSI')
		d = 0.9526475062940741
		fCA = oriA + YA * d
		# Projected angle and distance of current CA atom
		CA = self.pose.GetAtom(self.i, 'CA')
		T, P, _, d, radius = self.project(CA)
		fT, fP, _, fd, radius = self.project(fCA)
		###########################
		##### Reward Function #####
		###########################
		R = 0
		# R1 - Reward for moving forward
		if self.T > T: R += 1
		else:          R -= 1
		self.T = T
		# R2 - Penalty for distance from ellipse surface
		R -= 0.1 * d**2
		# R3 - Reward for being outside ellipse
		F1CA = np.linalg.norm(F1 - CA)
		F2CA = np.linalg.norm(F2 - CA)
		F1P = np.linalg.norm(F1 - P)
		F2P = np.linalg.norm(F2 - P)
		if F1CA > F1P and F2CA > F2P: R += 1
		else:                         R -= 1
		# R4 - Reward for going around ellipse clockwise
		if T < 180 and self.i != 0: self.switch = 1
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
		Ts, ds = defaultdict(list), defaultdict(list)
		for PHI in range(8):#self.action_space.high[0]):
			phi = self.get_angle_meanings(PHI)
			for PSI in range(8):#self.action_space.high[1]):
				psi = self.get_angle_meanings(PSI)
				fT, fP, fd, fr = self.future(phi=phi, psi=psi, F1=F1, F2=F2)
				Ts[fT].append(PHI)
				Ts[fT].append(PSI)
				ds[fd].append(PHI)
				ds[fd].append(PSI)
		min_fT_v  = min(Ts.keys())
		min_fT_aP = Ts[min_fT_v][0]
		min_fT_aS = Ts[min_fT_v][1]
		min_fd_v  = min(ds.keys())
		min_fd_aP = ds[min_fd_v][0]
		min_fd_aS = ds[min_fd_v][1]
		# Distance to C-term for loop closure
		C_term = np.linalg.norm(fCA - self.pose.GetAtom(0, 'C'))
		# Final features
		S = np.array([
			e, self.i, OE, T, d, self.switch,
			min_fT_aP, min_fT_aS, min_fT_v,
			min_fd_aP, min_fd_aS, min_fd_v,
			C_term])
		###########################
		### End State Condition ###
		###########################
		St = False
		# St1 - If polypeptide reaches max amino acids
		MAX = 15
		if self.i >= MAX:
			St = True
		# St2 - End game if the chain made a circle onto itself
		CAs   = [self.pose.GetAtom(x, 'CA') for x in range(self.i)]
		VECs  = [CA - fCA for CA in CAs]
		MAGs  = [np.linalg.norm(VEC) for VEC in VECs]
		CHECK = [1 if x < 1.5 else 0 for x in MAGs]
		if 1 in CHECK:
			St = True
			# Rt - Reward at this end state only
			R = self.i - MAX
		# End game if N-term to C-term distance < 1.5
		N_term = self.pose.GetAtom(0, 'N')
		C_term = self.pose.GetAtom(self.i, 'C')
		vNC = C_term - N_term
		distance = np.linalg.norm(vNC)
		if distance < 1.5:
			St = True
			R = len(self.pose.data['Amino Acids'])/MAX
		###########################
		####### Extra Info ########
		###########################
		info, data = None, {}
		return(S, R, St, info, data)

def play(show=True):
	''' Manually play the game '''
	print('\n' + '='*65)
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
	print('='*65)
	seed = input('\nChoose a seed value, empty for a random seed, q to quit > ')
	if    seed == '': seed = None
	elif  seed == 'q': exit()
	else: seed = int(seed)
	print('\n' + '-'*170)
	print(' '*55, 'Features', ' '*95, 'Reward')
	print('-'*160 + '|' + '-'*9)
	env = MolecularTetris()
	env.seed(seed)
	obs = env.reset()
	n = env.observation_space.shape[0] - 1
	types = ['e', 'i', 'OE', 'T', 'd', 'Switch',
			'Ta-phi', 'Ta-psi', 'fT',
			'da-phi', 'da-psi', 'fd',
			'C-term']
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
		inp = input('Actions [0-7, 0-7] no space, q to quit > ')
		if inp == 'q': exit()
		try: inp1 = int(inp[0]) ; inp2 = int(inp[1])
		except: print('incorrect input') ; continue
		if inp1 not in As: print('incorrect input') ; continue
		if inp2 not in As: print('incorrect input') ; continue
		inp = [inp1, inp2]
		if not isinstance(inp, list): print('incorrect input') ; continue
		actions.append([inp[0], inp[1]])
		obs = env.step([inp[0], inp[1]])
		R = round(obs[1], 3)
		output = ''
		for F in obs[0]: F = round(F, 1) ; output += '{:<{}}'.format(F, n)
		output += ' '*7 + '{:<5}'.format(R)
		print(output)
		Gt.append(R)
		St = obs[2]
	print('='*170)
	print('Actions:', actions)
	print('-'*20)
	print('Total Reward =', sum(Gt))
	if show: env.render()

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
	from tianshou.policy import PPOPolicy, DQNPolicy, BranchingDQNPolicy
	from tianshou.utils.net.discrete import Actor, Critic
	from tianshou.utils.net.common import ActorCritic, Net, BranchingNet
	from tianshou.env import DummyVectorEnv, SubprocVectorEnv
	from tianshou.data import Collector, VectorReplayBuffer, Batch
	from tianshou.trainer import onpolicy_trainer, offpolicy_trainer
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print(device)
	# 0. Get information about the environment
	env = MolecularTetris()
	features = env.observation_space.shape
	actions = env.action_space.shape
	# 1. Setup the training and testing environments
	train = SubprocVectorEnv([lambda:env for _ in range(350)])
	tests = SubprocVectorEnv([lambda:env for _ in range(150)])
	# 2. Setup neural networks and policy - PPO
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
			step_per_epoch=50000,
			max_epoch=epochs,
			step_per_collect=20,
			episode_per_test=10,
			repeat_per_collect=10,
			batch_size=256,
			stop_fn=lambda mean_reward: mean_reward >= 20)
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
