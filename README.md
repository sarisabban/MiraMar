# MolecularTetris
A game to build and cyclise a polypeptide for reinforcement learning 

![Alt Text](image.png)

## Description:
This is a game designed using OpenAI's gym library that builds a cyclic polypeptide molecule.

## How to play:
The goal is to build a cyclic polypeptide molecule, one amino acid at a time,going around the elliptical path.

The **actions** are as follows:
| Action   | Name | Values                                             |
|----------|------|----------------------------------------------------|
|Phi angle |P     |{0:0, 1:45, 2:90, 3:135, 4:180, 5:225, 6:270, 7:315}|
|Psi angle |S     |{0:0, 1:45, 2:90, 3:135, 4:180, 5:225, 6:270, 7:315}|

The **features** are as follows:
| Feature                                | Name | Values   | Description           |
|----------------------------------------|------|----------|-----------------------|
|Eccentricity                            |e     |[0, 1]    |Eccentricity of the ellipse|
|Index of step                           |i     |[0, 15]   |The index of the state step|
|Odd/Even of step                        |OE    |[0, 1]    |Whether the step is even or odd value|
|Angle of Cα                             |T     |[0, 360]  |The angle of the latest Cα atom from the start position through the centre of the ellipse|
|Distance of Cα                          |mag   |[-50, 50] |The distance of the Cα atom from the surface of the ellipse|
|Switch                                  |Switch|[0, 1]    |The point where the chain switchs from moving away from the start position the returning back|
|Phi angle action for lowest angle T     |Ta-phi|[0, 7]    |The phi action that will result in the greatest leap forward (most reduced angle)|
|Psi angle action for lowest angle T     |Ta-psi|[0, 7]    |The psi action that will result in the greatest leap forward (most reduced angle)|
|Resulting angle T                       |Tv    |[0, 360]  |The predicted angle that will result if the reccomended phi and psi actions were taken|
|Phi angle action for lowest distance mag|Da-phi|[0, 7]    |The phi action that will result in the least distance to the ellipse surface|
|Psi angle action for lowest distance mag|Da-psi|[0, 7]    |The psi action that will result in the least distance to the ellipse surface|
|Resulting distance mag                  |Dv    |[-50, 50] |The predicted distance that will result if the reccomended phi and psi actions were taken|
|Distance to C-term                      |C-term|[0, 1000] |The distance from N-term to C-term (for loop closure)|

The **rewards** are as follows:
| Reward                        | Name | Values                   | Description           |
|-------------------------------|------|--------------------------|-----------------------|
|Forward/Backward move          |R1    |±1                        |When current Cα angle is less than previous angle (moving forward) +1|
|Cα Distance                    |R2    |-0.1*distance<sup>2</sup> |Cα distance from ellipse surface (more negative further away)|
|Cα outside/inside ellipse      |R3    |±1                        |If the Cα is outside the ellipse +1|
|Moving clockwise/anti-clockwise|R4    |±1                        |If the Cα if moving away from the start poisition before the switch and towards the start position after the switch|
|Pre-mature end                 |Rt    |i - 20                    |If the peptide chain makes a circle around itself the game will end and a penalty is given, larger the chain the less the penalty|

## How to use:
`pip install numpy gym pytorch tainshou`

`git clone https://github.com/sarisabban/Pose.git`

`python3 MolecularTetris.py -p` to manually play the game. Follow on screen instructions. 

`python3 MolecularTetris.py -rl` to train a reinforcement learning agent (PPO using PyTorch and Tainshou).

`python3 MolecularTetris.py -rlp policy.pth` to have the reinforcement learning agent play the game using the *policy.pth* policy file.

The output of the game play are two .pdb (protein databank) files called molecule.pdb and path.pdb. These files can be viewed using PyMOL 'apt install pymol', or any other molecular visualisation software, or you can upload the structures [here](https://www.rcsb.org/3d-view) and view the files on a web browser.
