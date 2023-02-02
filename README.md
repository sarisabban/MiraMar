# MolecularTetris
A game to build and cyclise a polypeptide for reinforcement learning 

![Alt Text](image.png)

## Description:
This is a game designed using OpenAI's gym library that builds a cyclic polypeptide molecule.



## Dependencies:
`pip3 install numpy gym tainshou pytorch`

`git clone https://github.com/sarisabban/Pose.git`

## How to use:
The features are as follows:
| Feature                    | Value (float)| Description           |
|----------------------------|--------------|-----------------------|
|Eccentricity                | [0, 1]       |Eccentricity of the ellipse|
|Index of step               | [0, 15]      |The index of the state step|
|Odd/Even of step            | [0, 1]       |Whether the step is even or odd value|
|Angle of Cα                 | [0, 360]     |The angle of the latest Cα atom from the start position through the centre of the ellipse|
|Distance of Cα              | [-50.50]     |The distance of the Cα atom from the surface of the ellipse|
|Switch                      | [0, 1]       |The point when the amino acid switching from moving away from the start position the returning back|
|Reccomended angle action    | [0, 7]       |The action that will result in the greatest leap forward (most reduced angle)|
|Resulting angle             | [0, 360]     |The predicted angle that will result if the reccomended angle action was taken|
|Reccomended distance action | [0, 7]       |The action that will result in least distance to the ellipse surface|
|Resulting distance          | [-50, 50]    |The predicted that will result if the reccomended distance action was taken|

The Rewards are as follows:
| Reward                     | Value          | Description           |
|----------------------------|----------------|-----------------------|
|Forward move                | Int +1         |When current Cα angle is less than previous angle|
|Backward move               | Int -1         |When current Cα angle is larger than previous angle|
|Distance                    | Float -distance|Distance of Cα from the surface of the ellipse|
|Cα outside ellipse          | Int +1         |If the Cα is outside the ellipse|
|Cα inside ellipse           | Int -1         |If the Cα is inside the ellipse|
|Moving clockwise            | Int +1         |If the Cα if moving away from the start poisition before the switch and towards the start position after the switch|
|Moving anti-clockwise       | Int -1         |If the Cα if moving towards the start poisition before the switch and away from the start position after the switch|



`python3 MolecularTetris.py --play` to manually play the game

`python3 MolecularTetris.py --rl` to train a reinforcement learning agent (PPO using PyTorch and Tainshou)

`python3 MolecularTetris.py --rl policy.pth` to have the reinforcement learning agent play the game using the *policy.pth* policy file
