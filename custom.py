import math
import pymol
import numpy as np
from pymol.cgo import *

def RotationMatrix(thetaX, thetaY, thetaZ):
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

def point(selection):
	'''
	Prints the coordinate of a give selection, to be used to find the center
	of the path
		1. Select a residue in PyMOL
		2. Use the following command in the PyMOL terminal: center('sele')
	'''
	selection = selection + ' and n. CA'
	model = querying.get_model(selection)
	coords = model.get_coord_list()
	coords_matrix = np.array(coords)
	coords_ave = coords_matrix.mean(axis=0)
	coords_ave = coords_ave.tolist()
	coords_ave = [round(x, 1) for x in coords_ave]
	print('Coordinates of center of the path:', coords_ave)
	cmd.extend('Center', center)
	return(coords_ave)

def path(Cx, Cy, Cz, a, b, o, j, w):
	'''
	Draw the elliptical path given the parameters a, b, o, j, w
		1. Select a residue in PyMOL
		2. Use the command center('sele') to get starting Cx, Cy, Cz values
		3. Use the command path(Cx, Cy, Cz, a, b, o, j, w) to draw the path
			start with path(0, 0, 0, 4, 3, 0, 0, 0)
		4. Repeat the last command adjusting while the parameters
	'''
	try:
		pymol.cmd.delete('Center_')
		pymol.cmd.delete('Path')
	except: pass
	R = RotationMatrix(o, j, w)
	pymol.cmd.pseudoatom('Center_', pos=[Cx, Cy, Cz])
	pymol.cmd.show('spheres', 'Center_')
	pymol.cmd.set('sphere_scale', 0.25)
	C = [Cx, Cy, Cz]
	cr = abs(float(1.0))
	cg = abs(float(0.4))
	cb = abs(float(0.8))
	path = [BEGIN, LINES, COLOR, cr, cg, cb]
	UT, UB = [], []
	for x in np.arange(-a, a, 0.001):
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
		UT.append(ut)
		UB.append(ub)
	for ut in UT:
		path.append(VERTEX)
		path.append(ut[0])
		path.append(ut[1])
		path.append(ut[2])
	for ub in UB:
		path.append(VERTEX)
		path.append(ub[0])
		path.append(ub[1])
		path.append(ub[2])
	path.append(END)
	pymol.cmd.load_cgo(path, 'Path')
