# MiraMar
*De novo* cyclic protein polypeptide design using reinforcement learning.

<p align="center"><img src="logo.png" width="80%" height="80%" /></p>

## Description:
This is an environment designed to be compatible with OpenAI's gymnasium (not gym) library and optimised for Python 3.10+ that builds a cyclic protein polypeptide molecule. The goal is to build a cyclic polypeptide molecule, one amino acid at a time, going around an elliptical path, while hitting specific targets.

## How to setup:
Install the depedencies using this command:

```
pip install numpy scipy gymnasium git+https://github.com/sarisabban/Pose
```

The output of the environment play are two .pdb (protein databank) files called *molecule.pdb* and *path.pdb*. These files can be viewed using PyMOL `apt install pymol`, or any other molecular visualisation software, or you can upload these structures [here](https://www.rcsb.org/3d-view) and view the files on a web browser.

## How to play:
To play by code (standard gymnasium setup):

```py
from MiraMar import MiraMar

# Call the environment
env = MiraMar(render_mode='ansi')

# Information about the environment
observation_space = env.observation_space
action_space = env.action_space
metadata = env.metadata
render_mode = env.render_mode
reward_range = env.reward_range

# Reset the environment
observation, info = env.reset(seed=0)

# Take actions
actions = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(actions)
observation, reward, terminated, truncated, info = env.step(actions)

# See/Export the results
env.render() # See final result
env.export() # Export only the molecule
```
A step adds an amino acid and rotates its Φ and Ψ torsion angles as such `env.step([AMINO ACID, PHI, PSI])`.

You can use `env.render(show=False, save=True)` to save rather than show the environment output, you must have PyMOL installed to display the output. If no output is displayed, i.e PyMOL does not automatically open then go to the python venv's directory (python's virtual environment), open the *pyvenv.cfg* and change the line *include-system-site-packages = false* to *true* (all small letters). Understand that when rendering the ends of the molecule will be adjusted (2 hydrogens from the N-terminus and 1 oxygen from the C-terminus will be removed to facilitate a fused cyclic peptide).

To generate a **custom path** and **targets** use the following command: `observation, info = env.reset(custom=[[Cx, Cy, Cz], a, b, o, j, w, [[T1x, T1y, T1z], [T2x, T2y, T2z] ... ]])` for example `observation, info = env.reset(custom=[[3, 4, 5], 5, 4, 11, 12, 13, [[5, 6, 8], [3, 1, 4]]])` where *[Cx, Cy, Cz]* are the coordinates of the centeral point of the ellipse, *a* is the semi-major axis of the ellipse, *b* is the semi-minor axis of the ellipse, and *o*, *j*, *w* are the angle orientations of the ellipse with values 0° to 90°. Then finally the list of the target coordinates, *[T1x, T1y, T1z]* is the coordinates of target 1, *[T2x, T2y, T2z]* is the coordinates of target 2, etc. This command will automatically freeze the random seed to the value of 0.

## Environment details:
The **actions** are as follows:
| Action   | Name | Values                   |
|----------|------|--------------------------|
|Amino acid|AA    |Canonical and non-canonical D-amino acids and L-amino acids (52 actions)|
|Phi angle |P     |0-359 angles (360 actions)|
|Psi angle |S     |0-359 angles (360 actions)|

The **features** are as follows:
| Feature                             | Name    | Values   | Description           |
|-------------------------------------|---------|----------|-----------------------|
|Eccentricity                         |e        |[0, 1]    |Eccentricity of the ellipse|
|Index of step                        |i        |[0, 20]   |The index of the state step|
|Odd/Even of step                     |OE       |[0, 1]    |Whether the step is even or odd value|
|Angle of Cα                          |T        |[0, 360]  |The angle of the latest Cα atom from the start position through the centre of the ellipse|
|Distance of Cα                       |d        |[0, 100]  |The distance of the Cα atom from the surface of the ellipse|
|Switch                               |Switch   |[0, 1]    |The point where the chain switchs from moving away from the start position the returning|
|Φ angle for lowest distance to path  |fda-Φ    |[0, 360]  |The phi action that will result in the least distance to the ellipse surface|
|Ψ angle for lowest distance to path  |fda-Ψ    |[0, 360]  |The psi action that will result in the least distance to the ellipse surface|
|Φ angle for lowest distance to target|Tr_aP    |[0, 360]  |The phi action that will result in the least distance to the target|
|Ψ angle for lowest distance to target|Tr_aS    |[0, 360]  |The psi action that will result in the least distance to the target|
|Number of remaining target           |Trgs     |[3, 10]   |The number of remaining targets that goes down as targets are hit or missed|
|Direction of target                  |direction|[0, 1]    |The direction of the target wheather in the direction of the sidechain or away from it|
|Distance to C-term                   |C-term   |[0, 100]  |The distance from N-term to C-term (for loop closure)|

The **rewards** are as follows:
| Reward                        | Name        | Values                     | Description           |
|-------------------------------|-------------|----------------------------|-----------------------|
|Cα Distance                    |R            |$-{1 \over 9} d^2 + 1 + R_T$|Cα distance d from ellipse surface (intermediate reward shape) with $R_T = -{1 \over 33} SC + 1$ if hit, or 0 if far or no χ angle but used GLY amino acid, or -1 if far or no χ angle and used any amino acid other than GLY, or -1 for a miss, and where SC is the number of atoms in a side chain|
|Loop closure                   |R<sub>c</sub>|$-100 i + 2100 + F$         |If Sr3 condition is met for loop closure (actual episodic reward) with the F potential function as $F = {-100 \over 3.68} C + {500 \over 3.68}$ where C is the C-term|

The reward is +100 when the loop closes and the episode terminates, with R1 intermediate rewards shaped to assist the agent into finding the terminal (loop closing) state.

The **stop conditions** are as follows:
| Condition                     | Name | Reward Value | Description           |
|-------------------------------|------|--------------|-----------------------|
|Polypeptide length i=N         |St    |R             |Termination: when the polypeptide reachs a maximum length of N amino acids|
|Self loop                      |Sr1   |R             |Truncation: if the peptide chain makes a circle around itself the environment will end|
|Moving backwards               |Sr2   |R             |Truncation: if the T<sub>t</sub> < T<sub>t-1</sub>|
|Loop closure                   |Sr3   |R<sub>c</sub> |Truncation: if T<sub>t</sub> < 90 and the N-term to C-term distance is between 1.27 Å and 5 Å|

These termination and turnication conditions ensure that low rewards occure when sub-optimal molecules are built.

> __Note__
> 
> **i**: is the current index of the amino acid.
> 
> **N**: is largest possible size of a polypeptide allowed by the environment (20 amino acids).

## The information output:
When the environment is at the initial state right after `env.reset()`:
```py
info = {}
```

When the environment is not at the terminal state, the info is structured as follows:
```py
info =
	{
	'actions':[list of each step's actions],
	'rewards':[list of each step's rewards],
	'terminations':[list of each step's termination flag],
	'truncations':[list of each step's truncation flag]}
	}
```

When the environment is at the terminal state, the info is structured as follows:
```py
info = 
	{
	'actions':[list of each step's actions],
	'rewards':[list of each step's rewards],

	'terminations':[list of each step's termination flag],
	'truncations':[list of each step's truncation flag],
	'episode':
			{
			'r':Final return of episode,
			'l':Length of molecule,
			't':Time it took to complete this episode
			},
	'sequence':FASTA sequence of the molecule,
	'terminal_obs':Final observation as a list (not an array),
	'molecule':The pose.data JSON data structure with all of the molecule's information}
	}
```

## Custom Path and Targets (GUI with PyMOL):
To generate a path and targets to a specific molecular structure you can use the following steps:
1. Use the following command to open the PDB molecule in PyMOL and import the required functions: `pymol custom.py -b FILENAME.pdb`.
2. Select a residue where the center of the elliptical should be.
3. Run the following command `point('sele')` to get a starting Cx, Cy, Cz coordinate values of the center point of the elliptical path.
4. Run `path(Cx, Cy, Cz, a, b, o, j, w)` to generate the elliptical path.
5. Adjust the *Cx, Cy, Cz, a, b, o, j, w* values and run the last command again until the ideal path is generated.
6. Select residues to target and use the `point('sele')` command to get each target's coordinates.

## Training:
Provided is the `RL.py` script that trains a Proximal Policy Optimization (PPO) agent on the MiraMar environment or plays an already trained agent `agent.pth`. Instructions are isolated within the script itself, since this training process is separate from the actual environment setup. The following are the training results of that produced the *agent.pth* trained agent:
<p align="center">
	<img src="plots.png" width="40%" height="40%"> <img src="output.png" width="28%" height="28%" /></p>

## Post Generation Refinement:
Recomended to use [OSPREY3](https://github.com/donaldlab/OSPREY3) to refine the structure and make it more energetically favorable.
