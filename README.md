# MolecularTetris
A game to build cyclic polypeptides using reinforcement learning.
```
=============================================
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
           4                              
=============================================
```

![Alt Text](image.png)

## Description:
This is a game designed using OpenAI's gym library that builds a cyclic polypeptide molecule.

## How to use:
The goal is to build a cyclic polypeptide molecule, one amino acid at a time, going around an elliptical path.

`pip install numpy gym pytorch tainshou git+https://github.com/sarisabban/Pose`

`python3 MolecularTetris.py -p` to manually play the game. Follow on screen instructions. 

`python3 MolecularTetris.py -rl` to train a reinforcement learning agent (DQN using PyTorch and Tainshou).

`python3 MolecularTetris.py -rlp policy.pth` to have the reinforcement learning agent play the game using the *policy.pth* policy file.

The output of the game play are two .pdb (protein databank) files called *molecule.pdb* and *path.pdb*. These files can be viewed using PyMOL `apt install pymol`, or any other molecular visualisation software, or you can upload the structures [here](https://www.rcsb.org/3d-view) and view the files on a web browser.

To play by code (standard gym setup):

```
env = MolecularTetris()
print('Observation space:', env.observation_space)
print('Action space:', env.action_space)
env.seed(0)
env.reset()
env.step(4, [4, 4])
env.render() # env.render(show=False, save=True) to save rather than show the game output
```

The **actions** are as follows:
| Action   | Name | Values                   |
|----------|------|--------------------------|
|Amino Acid|AA    |Canonical and non-canonical D-amino acids and L-amino acids (52 actions)|
|Phi angle |P     |0-359 angles (360 actions)|
|Psi angle |S     |0-359 angles (360 actions)|

The **features** are as follows:
| Feature                                | Name | Values   | Description           |
|----------------------------------------|------|----------|-----------------------|
|Eccentricity                            |e     |[0, 1]    |Eccentricity of the ellipse|
|Index of step                           |i     |[0, 15]   |The index of the state step|
|Odd/Even of step                        |OE    |[0, 1]    |Whether the step is even or odd value|
|Angle of Cα                             |T     |[0, 360]  |The angle of the latest Cα atom from the start position through the centre of the ellipse|
|Distance of Cα                          |d     |[-50, 50] |The distance of the Cα atom from the surface of the ellipse|
|Switch                                  |Switch|[0, 1]    |The point where the chain switchs from moving away from the start position the returning back|
|Phi angle action for lowest angle T     |Ta-phi|[0, 7]    |The phi action that will result in the greatest leap forward (most reduced angle)|
|Psi angle action for lowest angle T     |Ta-psi|[0, 7]    |The psi action that will result in the greatest leap forward (most reduced angle)|
|Expected future angle T                 |fT    |[0, 360]  |The predicted angle that will result if the reccomended phi and psi actions were taken|
|Phi angle action for lowest distance mag|da-phi|[0, 7]    |The phi action that will result in the least distance to the ellipse surface|
|Psi angle action for lowest distance mag|da-psi|[0, 7]    |The psi action that will result in the least distance to the ellipse surface|
|Expected future distance mag            |fd    |[-50, 50] |The predicted distance that will result if the reccomended phi and psi actions were taken|
|Distance to C-term                      |C-term|[0, 1000] |The distance from N-term to C-term (for loop closure)|

The **rewards** are as follows:
| Reward                        | Name | Values                   | Description           |
|-------------------------------|------|--------------------------|-----------------------|
|Forward/Backward move          |R1    |±1                        |When current Cα angle is less than previous angle (moving forward) +1 reward|
|Cα Distance                    |R2    |-0.1*distance<sup>2</sup> |Cα distance from ellipse surface (more negative further away)|
|Cα outside/inside ellipse      |R3    |±1                        |If the Cα is outside the ellipse +1 rewards|
|Moving clockwise/anti-clockwise|R4    |±1                        |If the Cα if moving away from the start poisition before the switch and towards the start position after the switch|
|Target rewards                 |Rr    |+10 hit -10 miss -1 wrong AA 0 far|If the agent hits a target +10 reward, if failed to hit target because it chose the wrong amino acid -1 penalty, if it passed the target without hitting -10 penalty, if the target is too far away 0 reward|
|Pre-mature end                 |Rt    |i - N                     |If the peptide chain makes a circle around itself the game will end and a penalty is given, larger the chain the less the penalty|
|Loop closure                   |Rtc   |n / N                     |If N-term to C-term distance < 1.5 Å the game will end and a reward is given, shorter polypeptide give larger reward|

The **stop conditions** are as follows:
| Condition                     | Name | Values | Description           |
|-------------------------------|------|--------|-----------------------|
|Polypeptide length of i=N      |St1   |0       |The polypeptide can only reach a maximum length of N amino acids|
|Self circle                    |St2   |Rt      |If the peptide chain makes a circle around itself the game will end and a penalty is given, larger the chain the less the penalty|
|Loop closure                   |St3   |Rtc     |If the N-term to C-term distance < 1.5 Å|

> __Note__
> 
> **i**: is the current index of the amino acid
> 
> **n**: is current the final size of the built polypeptide.
> 
> **N**: is largest size of a polypeptide allowed by the game (20 amino acids).
