import math
import shutil
import warnings
import subprocess
import numpy as np
from Pose.pose import *
from gym.spaces import Dict, Discrete, Box, Tuple, MultiDiscrete
warnings.filterwarnings('ignore')

class MolecularTetris():
	''' Game for designing cyclic peptides using reinforcement learning '''
	def __init__(self):
		''' Initialise global variables '''
		self.observation_space = Box(
			low=np.array( [0,   0,   0, -50]),
			high=np.array([9, 360, 360,  50]),
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
		R = Rz.dot(Ry).dot(Rx)
		return(R)
	def path(self, export=False):
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
		if export:
			self.points = []
			self.points.append(C)
			self.points.append(F1)
			self.points.append(F2)
			for x in np.arange(-a, a, 1):
				y2 = (1 - x**2/a**2) * b**2
				yt =  math.sqrt(y2)
				yb = -math.sqrt(y2)
				ut = np.array([C[0], C[1], C[2]]) - np.array([x, yt, 0])
				ub = np.array([C[0], C[1], C[2]]) - np.array([x, yb, 0])
				ut = C - ut
				ub = C - ub
				ut = np.matmul(ut, R)
				ub = np.matmul(ub, R)
				ut = C + ut
				ub = C + ub
				pt = np.array([ut[0], ut[1], ut[2]])
				pb = np.array([ub[0], ub[1], ub[2]])
				self.points.append(pt)
				self.points.append(pb)
		return(start, F1, F2)
	def project(self, atom):
		''' Project an atom onto the ellipse to get point P and its angle T '''
		C, a, b = self.C, self.a, self.b
		R   = self.RotationMatrix(self.o, self.j, self.w)
		R_  = np.linalg.inv(R)
		atom_ = atom - C
		atom_ = np.matmul(atom_, R_)
		T   = math.atan2(atom_[1], atom_[0])
		r  = (a*b)/math.sqrt(a**2 * math.sin(T)**2 + b**2 * math.cos(T)**2)
		P_ = np.array([r * math.cos(T), r * math.sin(T), 0])
		P  = np.matmul(P_, R)
		P  = C + P
		vP = P - atom
		vP_mag = np.linalg.norm(vP)
		T   = T * 180 / math.pi
		if T < 0: T += 360.0
		if T == 0.0: T = 360.0
		return(T, P, vP, vP_mag, r)
	def render(self, show=False, save=True, filename='molecule'):
		''' Export the molecule as a PDB file and display it'''
		start, F1, F2 = self.path(export=True)
		T, P, vP, vP_mag, r = self.project(self.pose.GetAtom(self.i, 'CA'))
		self.points.append(P)
		fCA, vcfCA, centre = self.futureCA(vector=True)
		self.points.append(fCA)
		self.points.append(centre)
		T, P, vP, vP_mag, r = self.project(self.futureCA())
		self.points.append(P)
		Tx, Ty, Tz, X, Y, Z = self.angles(P - centre, axis=True)
		self.points.append(X + centre)
		self.points.append(Y + centre)
		self.points.append(Z + centre)
		with open('path.pdb', 'w') as F:
			for i, p in enumerate(self.points):
				if i == 0:
					a, e, c = 'C', 'C', 'A'
					F.write('HEADER Path Centre\n')
				elif i == 1:
					a, e, c = 'O', 'O', 'A'
					F.write('HEADER F1\n')
				elif i == 2:
					a, e, c = 'O', 'O', 'A'
					F.write('HEADER F2\n')
				elif i == 3:
					a, e, c = 'H', 'H', 'B'
					F.write('HEADER Path\n')
				elif i == len(self.points)-7:
					a, e, c = 'I', 'I', 'C'
					F.write('HEADER Projection of Current CA\n')
				elif i == len(self.points)-6:
					a, e, c = 'S', 'S', 'C'
					F.write('HEADER Future CA\n')
				elif i == len(self.points)-5:
					a, e, c = 'O', 'O', 'C'
					F.write('HEADER Axis Centre\n')
				elif i == len(self.points)-4:
					a, e, c = 'S', 'S', 'C'
					F.write('HEADER Projected of Future CA\n')
				elif i == len(self.points)-3:
					a, e, c = 'H', 'H', 'C'
					F.write('HEADER X-axis\n')
				elif i == len(self.points)-2:
					a, e, c = 'H', 'H', 'C'
					F.write('HEADER Y-axis\n')
				elif i == len(self.points)-1:
					a, e, c = 'H', 'H', 'C'
					F.write('HEADER Z-axis\n')
				else:
					a, e, c = 'H', 'H', 'B'
				A, l, r, s, I = 'ATOM', '', 'GLY', 1, ''
				x, y, z, o, t, q = p[0], p[1], p[2], 1.0, 1.0, 0.0
				Entry = self.pose.PDB_entry(A,i+1,a,l,r,c,s,I,x,y,z,o,t,q,e)
				F.write(Entry)
		self.pose.Export('{}.pdb'.format(filename))
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
	def position(self):
		''' Position the polypeptide at the start position of the path '''
		CA = self.pose.GetAtom(0, 'CA')
		AAcoord = self.pose.data['Coordinates']
		AAcoord = (self.start - CA) + AAcoord
		Ax, Ay, Az = 180+self.o,  0-self.j,  -45-self.w
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
		self.start, F1, F2 = self.path()
		self.addAA()
		self.T = 360
		S, R, St, info, data = self.SnR(self.start, F1, F2)
		return(S, data)
	def step(self, action):
		''' Play one step, add amino acid and define its phi/psi angles '''
		AA  = 'G'
		phi = self.get_angle_meanings(action)
		psi = None
		self.addAA(AA, phi, psi)
		self.i = max(self.pose.data['Amino Acids'].keys())
		start, F1, F2, = self.path()
		return(self.SnR(start, F1, F2))
	def futureCA(self, vector=False):
		''' Get future CA's position given current amino acid '''
		N  = self.pose.GetAtom(self.i, 'N')
		CA = self.pose.GetAtom(self.i, 'CA')
		C  = self.pose.GetAtom(self.i, 'C')
		vNCA = N - CA
		vCAC = CA - C
		vNCA_mag = np.linalg.norm(vNCA)
		vNCA = vNCA*(3.716521828269005/vNCA_mag)
		vNCA_mag = np.linalg.norm(vNCA)
		centre = N - vNCA
		vNCAxvCAC = np.cross(vNCA, vCAC)
		vNCAxvNCAxvCAC = np.cross(vNCA, vNCAxvCAC)
		vNCAxvNCAxvCAC_mag = np.linalg.norm(vNCAxvNCAxvCAC)
		vNCAxvNCAxvCAC = vNCAxvNCAxvCAC*(3.18706212678699/vNCAxvNCAxvCAC_mag)
		fCA = vNCAxvNCAxvCAC + centre
		vcfCA = centre - fCA
		if vector:
			return(fCA, vcfCA, centre)
		else:
			return(fCA)
	def angles(self, v, axis=False):
		''' Get direction angles that point towards a projection point '''
		N  = self.pose.GetAtom(self.i, 'N')
		CA = self.pose.GetAtom(self.i, 'CA')
		C  = self.pose.GetAtom(self.i, 'C')
		vX = CA - N
		vCAC = CA - C
		X = (vX / np.linalg.norm(vX))
		X_mag = np.linalg.norm(X)
		vZ = np.cross(vX, vCAC)
		Z = (vZ / np.linalg.norm(vZ))
		Z_mag = np.linalg.norm(Z)
		vY = np.cross(vX, vZ)
		Y = (vY / np.linalg.norm(vY))
		Y_mag = np.linalg.norm(Y)
		v_mag = np.linalg.norm(v)
		Tx = math.acos(np.dot(v, X)/(v_mag)) * 180/math.pi
		Ty = math.acos(np.dot(v, Y)/(v_mag)) * 180/math.pi
		Tz = math.acos(np.dot(v, Z)/(v_mag)) * 180/math.pi
		AxX = np.cross(v, X)
		AxY = np.cross(v, Y)
		AxZ = np.cross(v, Z)
		if AxX[0] <= 0: Tx = 360 - Tx
		if AxY[1] <= 0: Ty = 360 - Ty
		if AxZ[2] <= 0: Tz = 360 - Tz
		if axis:
			return(Tx, Ty, Tz, X, Y, Z)
		else:
			return(Tx, Ty, Tz)
	def SnR(self, start, F1, F2):
		''' Return the state features and rewards after each game step '''
		fCA, _, centre = self.futureCA(vector=True)
		T, P, vP, vP_mag, r = self.project(fCA)
		Tx, Ty, Tz = self.angles(P - centre)
		if self.i == 0: Tz = 90.0
		# Reward function
		R = 0
		if T < self.T: R += 1
		else: R -= 1
		self.T = T
		# Features
		if   (self.i % 2) != 0: OE = 0
		elif (self.i % 2) == 0: OE = 1
		S = np.array([self.i, T, Tz, vP_mag])
		# End state condition
		St = False
		if self.i >= 15:
			R = 1
			St = True
		# Extra info
		info, data = None, {}
		return(S, R, St, info, data)

def play(show=True):
	''' Manually play the game '''
	print('''\
	╔╦╗┌─┐┬  ┌─┐┌─┐┬ ┬┬  ┌─┐┬─┐  ╔╦╗┌─┐┌┬┐┬─┐┬┌─┐
	║║║│ ││  ├┤ │  │ ││  ├─┤├┬┘   ║ ├┤  │ ├┬┘│└─┐
	╩ ╩└─┘┴─┘└─┘└─┘└─┘┴─┘┴ ┴┴└─   ╩ └─┘ ┴ ┴└─┴└─┘
	           0
	
	   7          /    1
	             /
	            /
	6          0          2
	
	        ACTIONS
	   5               3
	
	           4
	
	Feature 1: Index: of step (game has max 15 steps)
	Feature 2: Angle: +1 reward if it decreases, -1 reward if it increases
	Feature 3: Direction: angle to point
	Feature 4: Distance: to point''')
	seed = input('Choose a seed value, empty for a random seed, q to quit > ')
	if seed == '': seed = None
	elif seed == 'q': exit()
	else: seed = int(seed)
	print('-'*80)
	print(' '*15, 'Features', ' '*40, 'Reward')
	print('-'*55 + '|' + '-'*24)
	env = MolecularTetris()
	env.seed(seed)
	obs = env.reset()
	index = int(obs[0][0])
	angle = round(obs[0][1], 3)
	dirct = round(obs[0][2], 3)
	disnt = round(obs[0][3], 3)
	output = '{:<15}{:<15}{:<15}{:<15}' \
	.format(index, angle, dirct, disnt)
	print(output)
	Gt = []
	St = False
	while(St==False):
		inp = input('Action > ')
		if inp == 'q': exit()
		try:
			inp = int(inp)
		except:
			print('incorrect input')
			continue
		if inp not in [0, 1, 2, 3, 4, 5, 6, 7]:
			print('incorrect input')
			continue
		obs = env.step(inp)
		index = int(obs[0][0])
		angle = round(obs[0][1], 3)
		dirct = round(obs[0][2], 3)
		disnt = round(obs[0][3], 3)
		R = round(obs[1], 3)
		output = '{:<15}{:<15}{:<15}{:<15}{:<5}' \
		.format(index, angle, dirct, disnt, R)
		print(output)
		Gt.append(R)
		St = obs[2]
	if show: env.render(show=True, save=False)
	print('-----------------')
	print('Total Reward =', sum(Gt))

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

def RL(epochs):
	''' Reinforcement Learning '''
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
	train = SubprocVectorEnv([lambda:env for _ in range(100)])
	tests = SubprocVectorEnv([lambda:env for _ in range(75)])
	# 2. Setup neural networks and policy
	#---------- PPO
	net = Net(features, hidden_sizes=[64, 64, 64], device=device)
	actor = Actor(net, actions, device=device).to(device)
	critic = Critic(net, device=device).to(device)
	actor_critic = ActorCritic(actor, critic)
	optim = torch.optim.Adam(actor_critic.parameters(), lr=1e-4)
	dist = torch.distributions.Categorical
	policy = PPOPolicy(actor, critic, optim, dist, action_space=env.action_space, deterministic_eval=True)
#	#---------- DQN
#	net = Net(features, hidden_sizes=[128, 128, 128], device=device)
#	actor = Actor(net, actions, device=device).to(device)
#	optim = torch.optim.Adam(actor.parameters(), lr=1e-4)
#	policy = DQNPolicy(actor, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320)
#	#----------
	# 3. Setup vectorised replay buffer
	VRB = VectorReplayBuffer(20000, len(train))
	# 4. Setup collectors
	train_collector = Collector(policy, train, VRB)
	tests_collector = Collector(policy, tests)
	# 5. Train
	result = onpolicy_trainer(
#	result = offpolicy_trainer(
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
	torch.save(policy.state_dict(), 'policy.pth')
	# 7. Play
	tests = DummyVectorEnv([lambda:MolecularTetris() for _ in range(1)])
	tests_collector = Collector(policy, tests)
	policy.load_state_dict(torch.load('policy.pth'))
	policy.eval()
	result = tests_collector.collect(n_episode=1, render=True)
	print('Final reward: {}, length: {}'.format(result['rews'].mean(), result['lens'].mean()))



play()
#RL(4)
