#!/usr/bin/env python3

import math
import time
import scipy
import shutil
import pathlib
import warnings
import datetime
import subprocess
import numpy as np
import gymnasium as gym
from pose import *
warnings.filterwarnings('ignore')

class MiraMar():
	''' Game for designing cyclic peptides using reinforcement learning '''
	metadata = {'render_modes':['ansi', 'human']}
	def __init__(self, render_mode='ansi'):
		''' Initialise global variables '''
		self.bins = 360
		self.observation_space = gym.spaces.Box(
			low=np.array( [0,  0, 0,  0,  0, 0,  0,  0,  0,  0,  0,  3, 0,  0]),
			high=np.array([1, 20, 1,360,100, 1,360,360,100,360,360, 10, 1, 50]))
		self.action_space = gym.spaces.MultiDiscrete([52, self.bins, self.bins])
		self.reward_range = (-np.inf, np.inf)
		self.render_mode  = render_mode
		self.seed = None
	def get_angle_meanings(self, action):
		''' Definition of each action's angle '''
		angles = {a: 360/self.bins * a for a in range(self.bins)}
		return(angles[action])
	def get_residue_meanings(self, action):
		''' Definition of each action's residue '''
		residues = {
		0 :'A',  1:'B',  2:'C',  3:'D',  4:'E',  5:'F',  6:'G',  7:'H',  8:'I',
		9 :'J', 10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R',
		18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z',
		26:'a', 27:'b', 28:'c', 29:'d', 30:'e', 31:'f', 32:'g', 33:'h', 34:'i',
		35:'j', 36:'k', 37:'l', 38:'m', 39:'n', 40:'o', 41:'p', 42:'q', 43:'r',
		44:'s', 45:'t', 46:'u', 47:'v', 48:'w', 49:'x', 50:'y', 51:'z'}
		return(residues[action])
	def render(self, show=True, save=False, path=True, filename='molecule'):
		''' Export the molecule as a PDB file and display it '''
		points = self.path(path=True)
		if path:
			with open('path.pdb', 'w') as F:
				for i, p in enumerate(points):
					if   i == 0: a,e,c = 'N','N','A' ; F.write('HEADER C\n')
					elif i == 1: a,e,c = 'S','S','A' ; F.write('HEADER F1\n')
					elif i == 2: a,e,c = 'S','S','A' ; F.write('HEADER F2\n')
					elif i == 3: a,e,c = 'H','H','B' ; F.write('HEADER Path\n')
					else:        a,e,c = 'H','H','B'
					A, l, r, s, I = 'ATOM', '', 'GLY', 1, ''
					x, y, z, o, t, q = p[0], p[1], p[2], 1.0, 1.0, 0.0
					Entry = self.pose.PDB_entry(A,i+1,a,l,r,c,s,I,x,y,z,o,t,q,e)
					F.write(Entry)
			with open('path.pdb', 'a') as F:
				F.write('HEADER Targets\n')
				for i, point in enumerate(self.targetLST):
					p = point[1]
					a, e, c = 'O', 'O', 'C'
					A, l, r, s, I = 'ATOM', '', 'GLY', 1, ''
					x, y, z, o, t, q = p[0], p[1], p[2], 1.0, 1.0, 0.0
					Entry = self.pose.PDB_entry(A,i+1,a,l,r,c,s,I,x,y,z,o,t,q,e)
					F.write(Entry)
		self.pose.Export('{}.pdb'.format(filename))
		locate = shutil.which('pymol')
		if pathlib.Path('points.pdb').exists():
			display = ['pymol', 'molecule.pdb', 'path.pdb', 'points.pdb']
			removes = ['rm', 'molecule.pdb', 'path.pdb', 'points.pdb']
		else:
			display = ['pymol', 'molecule.pdb', 'path.pdb']
			removes = ['rm', 'molecule.pdb', 'path.pdb']
		if show == True and save == True:
			if locate == None: print('PyMOL not installed') ; return
			subprocess.run(display, capture_output=True)
		elif show == True and save == False:
			if locate == None: print('PyMOL not installed') ; return
			subprocess.run(display, capture_output=True)
			subprocess.run(removes)
		elif show == False and save == True: return
		elif show == False and save == False: subprocess.run(removes)
	def show(self, P, atom, define):
		''' Show a specific point in a new points.pdb file '''
		with open('points.pdb', 'a') as F:
			F.write(f'HEADER {define}\n')
			a, e, c = atom, atom, 'C'
			A, l, r, s, I = 'ATOM', '', 'GLY', 1, ''
			x, y, z, o, t, q = P[0], P[1], P[2], 1.0, 1.0, 0.0
			Entry = self.pose.PDB_entry(A,1,a,l,r,c,s,I,x,y,z,o,t,q,e)
			F.write(Entry)
	def export(self):
		''' Export the molecule only '''
		self.render(show=False, save=True, path=False)
	def close(self):
		''' Close and clear the environemnt '''
		self.pose = None
		return None
	def RotationMatrix(self, thetaX, thetaY, thetaZ):
		''' Rotation Matrix '''
		sx = math.sin(math.radians(thetaX))
		cx = math.cos(math.radians(thetaX))
		sy = math.sin(math.radians(thetaY))
		cy = math.cos(math.radians(thetaY))
		sz = math.sin(math.radians(thetaZ))
		cz = math.cos(math.radians(thetaZ))
		Rx = np.array([[  1,  0,  0], [  0, cx,-sx], [  0, sx, cx]])
		Ry = np.array([[ cy,  0, sy], [  0,  1,  0], [-sy,  0, cy]])
		Rz = np.array([[ cz,-sz,  0], [ sz, cz,  0], [  0,  0,  1]])
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
	def addAA(self, AA='M', phi=None, psi=None):
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
	def targetS(self, plot=False):
		''' Randomly generate target points the side chains should reach '''
		number = np.random.randint(3, 10)
		points = []
		for p in range(number):
			x = np.random.randint(-1e4, 1e4)
			y = np.random.randint(-1e4, 1e4)
			z = np.random.randint(-1e4, 1e4)
			v = np.array([x, y, z])
			mag = np.linalg.norm(v)
			nrm = v/mag
			distance = np.random.randint(self.a + 1, self.a + 5)
			point = nrm * distance
			point = self.C - point
			T, P, vP, vP_mag, r = self.project(point)
			points.append((T, point))
			if plot: self.show(point, 'O', f'target_{p}')
		points.sort(key=lambda x: x[0], reverse=True)
		self.targetLST = points
	def AminoAcidOri(self, ori='phi'):
		''' Get amino acid origin and axis from the phi or psi perspective '''
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
			OO = self.pose.GetAtom(self.i,'O') - self.pose.GetAtom(self.i,'OXT')
			vY = np.cross(OO, X)
			Y = vY / np.linalg.norm(vY)
			Y_mag = np.linalg.norm(Y)
			vZ = np.cross(Y, X)
			Z = vZ / np.linalg.norm(vZ)
			Z_mag = np.linalg.norm(Z)
			return(origin, X, Y, Z)
	def future(self, phi_psi, plot=False):
		''' For phi and psi angles return the future angles and points '''
		a, b = self.a, self.b
		phi, psi = math.radians(phi_psi[0]), math.radians(phi_psi[1])
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
		CA  = self.pose.GetAtom(self.i, 'CA')
		C   = self.pose.GetAtom(self.i, 'C')
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
		fT, fP, _, fd, fr = self.project(fCA_psi)
		if len(self.targetLST) != 0: ft=np.linalg.norm(fCA-self.targetLST[0][1])
		else: ft = 0
		if plot:
			self.show(fCA_psi, 'S', f'fCA_{phi_psi[0]}_{phi_psi[1]}')
			self.show(fP, 'I', f'fP_{phi_psi[0]}_{phi_psi[1]}')
		if   self.future_output == 'fT': return(fT)
		elif self.future_output == 'fd': return(fd)
		elif self.future_output == 'ft': return(ft)
	def fT_fd_ft(self):
		''' Determine which phi psi angles leads to lowest fT, fd or ft '''
		results = []
		for output in ['fT', 'fd', 'ft']:
			self.future_output = output
			solution = scipy.optimize.minimize(
				self.future,
				(72, 221),
				bounds=((0.00, 359.99), (0.00, 359.99)),
				method='SLSQP')
			results.append(solution.x[0])
			results.append(solution.x[1])
			results.append(solution.fun)
		fT_aP, fT_aS, fT_v = results[0], results[1], results[2]
		fd_aP, fd_aS, fd_v = results[3], results[4], results[5]
		Tr_aP, Tr_aS, Tr_v = results[6], results[7], results[8]
		return(fT_aP, fT_aS, fT_v, fd_aP, fd_aS, fd_v, Tr_aP, Tr_aS, Tr_v)
	def chi(self, chis):
		''' Rotate all chi angels and measure distance to target '''
		for c, v in enumerate(chis): self.pose.Rotate(self.i, v, 'CHI', c+1)
		edge = self.pose.data['Coordinates'][-4]
		distance = np.linalg.norm(self.target - edge)
		return(distance)
	def target_logic(self, AA):
		''' Rotating side chains and hitting targets '''
		hit, direction, CA_t, C_t = 0, 0, 0, 0
		Trgs = len(self.targetLST)
		if Trgs != 0:
			CA = self.pose.GetAtom(self.i, 'CA')
			C  = self.pose.GetAtom(self.i, 'C')
			T, P, _, d, radius = self.project(CA)
			if self.mark == True:
				self.targetLST.pop(0)
				self.mark = False
				hit = 3
			Trgs = len(self.targetLST)
			if Trgs != 0:
				self.target = self.targetLST[0][1]
				tT = self.targetLST[0][0]
				CA_t = np.linalg.norm(CA - self.target)
				C_t = np.linalg.norm(C - self.target)
				if tT > T and self.i != 0: self.mark = True
				if CA_t < C_t: direction = 1
				CHIs = len(self.pose.AminoAcids[AA]['Chi Angle Atoms'])
				if CA_t <= 13.0 and direction == 1 and CHIs != 0 and AA != 'P':
					x0 = tuple([180 for x in range(CHIs)])
					bs = tuple([(0.00, 359.00) for x in range(CHIs)])
					solution = scipy.optimize.minimize(
						self.chi, x0, bounds=bs, method='SLSQP')
					distance = solution.fun
					if 0 < distance < 3.3: hit = 1
					else: hit = 2
				elif CHIs == 0 or AA == 'P':
					edge = self.pose.data['Coordinates'][-4]
					distance = np.linalg.norm(self.target - edge)
					if 0 < distance < 3.3: hit = 1
					else: hit = 2
				if hit == 1:
					self.targetLST.pop(0)
					self.mark = False
		return(hit, Trgs, direction, CA_t)
	def reset(self, seed=None, custom=[]):
		''' Reset game '''
		self.time_start = time.time()
		self.pose = None
		self.step_rewards = []
		self.step_actions = []
		self.terms = []
		self.trncs = []
		if self.seed == None: self.seed = seed
		np.random.seed(self.seed)
		self.i = 0
		self.C = np.random.uniform(0, 50, size=(3,))
		self.b = np.random.uniform(4, 8)
		self.a = np.random.uniform(self.b, 10)
		self.o = np.random.uniform(0, 90)
		self.j = np.random.uniform(0, 90)
		self.w = np.random.uniform(0, 90)
		if custom != []:
			self.seed = 0
			np.random.seed(self.seed)
			self.C = custom[0]
			self.a = custom[1]
			self.b = custom[2]
			self.o = custom[3]
			self.j = custom[4]
			self.w = custom[5]
		self.start, F1, F2, e = self.path()
		self.addAA()
		self.targetS()
		self.T, self.F1P, self.switch, self.mark = 360, 0, 0, False
		S, R, St, Sr, info = self.SnR(self.start, F1, F2, e, 'M')
		return(S, info)
	def step(self, actions):
		''' Play one step, add an amino acid and define its phi/psi angles '''
		AA  = self.get_residue_meanings(actions[0])
		phi = self.get_angle_meanings(actions[1])
		psi = self.get_angle_meanings(actions[2])
		self.addAA(AA, phi, psi)
		self.i = max(self.pose.data['Amino Acids'].keys())
		start, F1, F2, e = self.path()
		self.step_actions.append(list(actions))
		return(self.SnR(start, F1, F2, e, AA.upper()))
	def SnR(self, start, F1, F2, e, AA):
		''' Return the state features and rewards after each game step '''
		if self.i == 0: T = 360
		# Maximum possible size of polypeptide
		MAX = 20
		# Calculating future CA
		oriA, XA, YA, ZA = self.AminoAcidOri(ori='PHI')
		fCA = oriA + YA * 3.1870621267869894
		# Projected angles and distances of current and future CA atoms
		N  = self.pose.GetAtom(self.i, 'N')
		CA = self.pose.GetAtom(self.i, 'CA')
		C  = self.pose.GetAtom(self.i, 'C')
		T, P, _, d, radius = self.project(CA)
		fT, fP, _, fd, radius = self.project(fCA)
		# Target logic
		hit, Trgs, direction, CA_t = self.target_logic(AA)
		SC_size = len(self.pose.AminoAcids[AA]['Vectors'])
		###########################
		##### Reward Function #####
		###########################
		R = 0.0
		# Path reward
		if self.i != 0: R = -(1/9) * d**2 + 1
		# Target reward
		if   hit == 1: R += (-1/33)*SC_size + 1           # Hit
		elif AA == 'G' and (hit == 0 or hit == 2): R += 0 # Far or no rotamers + G
		elif hit == 0 or hit == 2: R -= 1                 # Far or no rotamers
		elif hit == 3: R -= 1                             # Miss
		###########################
		######## Features #########
		###########################
		# Check if step is odd or even
		if   (self.i % 2) != 0: OE = 0
		elif (self.i % 2) == 0: OE = 1
		# Determine lowest fT and lowest fd
		fT_aP,fT_aS,fT_v,fd_aP,fd_aS,fd_v,Tr_aP,Tr_aS,Tr_v = self.fT_fd_ft()
		# Distance from N-term to C-term for loop closure
		C_term = np.linalg.norm(C - self.pose.GetAtom(0, 'N'))
		# The switch
		start_F2 = np.linalg.norm(F2 - self.pose.GetAtom(0, 'C'))
		if T < 180 and self.i != 0 and C_term > start_F2: self.switch = 1
		# Final features
		S = np.array([
			e, self.i, OE, T, d, self.switch,
			fd_aP, fd_aS, C_term,
			Tr_aP, Tr_aS, Trgs, direction, CA_t])
		############################
		### End State Conditions ###
		############################
		St, Sr = False, False
		# St - If polypeptide reaches max amino acids
		if self.i >= MAX: St = True
		# Sr1 - End game if the chain made a circle onto itself
		CAs   = [self.pose.GetAtom(x, 'CA') for x in range(self.i)]
		VECs  = [CA - fCA for CA in CAs]
		MAGs  = [np.linalg.norm(VEC) for VEC in VECs]
		CHECK = [1 if x < 2.0 else 0 for x in MAGs]
		if 1 in CHECK: Sr = True
		# Sr2 - End game if Tt < Tt-1
		if self.T < T: Sr = True
		# Sr3: loop closure
		if T < 90 and self.i > 5 and self.switch == 1 and 1.27 < C_term < 5:
			Sr = True
			F = -(100/3.68)*C_term + (500/3.68) # potential function
			R = -100*self.i + 2100 + F
		if self.i != 0: self.T = T
		###########################
		####### Information #######
		###########################
		if self.i != 0:
			self.step_rewards.append(R)
			self.terms.append(St)
			self.trncs.append(Sr)
		if St or Sr:
			self.time_end = time.time()
			finish_seconds = self.time_end - self.time_start
			finish_time = datetime.timedelta(seconds=finish_seconds)
			info = {
				'actions':self.step_actions,
				'rewards':self.step_rewards,
				'terminations':self.terms,
				'truncations':self.trncs,
				'episode':{
					'r':sum(self.step_rewards),
					'l':self.i,
					't':str(finish_time)},
				'sequence':self.pose.data['FASTA'],
				'terminal_obs':list(S),
				'molecule':self.pose.data}
		else:
			info = {
				'actions':self.step_actions,
				'rewards':self.step_rewards,
				'terminations':self.terms,
				'truncations':self.trncs}
		#########################
		####### Rendering #######
		#########################
		if self.render_mode == 'human':
			self.render(show=False, save=True)
			display = ['pymol', 'molecule.pdb', 'path.pdb']
			remove  = ['rm', 'molecule.pdb', 'path.pdb']
			subprocess.run(display, capture_output=True)
			subprocess.run(remove,  capture_output=True)
		return(S, R, St, Sr, info)
