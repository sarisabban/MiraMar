# MiraMar
*De novo* cyclic protein polypeptide design using reinforcement learning.

<p align="center"><img src="image.png" width="80%" height="80%" /></p>

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

You can use `env.render(show=False, save=True)` to save rather than show the environment output, must have PyMOL installed to display the output. If no output is displayed, i.e PyMOL does not automatically open then go to the venv's directory (python's virtual environment), open the *pyvenv.cfg* and change the line *include-system-site-packages = false* to be true (all small letters).

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
|Φ angle for lowest distance          |fda-Φ    |[0, 360]  |The phi action that will result in the least distance to the ellipse surface|
|Ψ angle for lowest distance          |fda-Ψ    |[0, 360]  |The psi action that will result in the least distance to the ellipse surface|
|Distance to C-term                   |C-term   |[0, 100]  |The distance from N-term to C-term (for loop closure)|

The **rewards** are as follows:
| Reward                        | Name | Values                             | Description           |
|-------------------------------|------|------------------------------------|-----------------------|
|Cα Distance                    |R     |(-2/71.7<sup>2</sup>)d<sup>2</sup>+1|Cα distance from ellipse surface|
|Loop closure                   |R     |+100                                |If Sr3 condition is met for loop closure|

The **stop conditions** are as follows:
| Condition                     | Name | Reward Value | Description           |
|-------------------------------|------|--------------|-----------------------|
|Polypeptide length of i=N      |St    |0             |Termination: when the polypeptide reachs a maximum length of N amino acids|
|Self loop                      |Sr1   |0             |Truncation: if the peptide chain makes a circle around itself the environment will end|
|Moving backwards               |Sr2   |0             |Truncation: if the T<sub>t</sub> < T<sub>t-1</sub>|
|Loop closure                   |Sr3   |+100          |Truncation: if the N-term to C-term distance between 0.0 and 3 Å|

> __Note__
> 
> **i**: is the current index of the amino acid.
> 
> **N**: is largest possible size of a polypeptide allowed by the environment (20 amino acids).

## The information output:
When the environment is at the initial state right after `env.reset()`:
```
info = {}
```

When the environment is not at the terminal state, the info is structured as follows:
```
info =
	{
	'actions':[list of each step's actions],
	'rewards':[list of each step's rewards],
	'terminations':[list of each step's termination flag],
	'truncations':[list of each step's truncation flag]}
	}
```

When the environment is at the terminal state, the info is structured as follows:
```
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

## Training:
Provided is the `RL.py` script that trains a PPO agent on the MiraMar environment or plays an already trained agent `agent.pth`. Instructions are isolated within the script itself, since this training process is separate from the actual environment setup.
