import utilFunctions
import perceptronFace
import os
import numpy as np

'''
Dimensions:
Faces: 70 x 60 - Begin with an empty line, end with empty line 
Digits: 28 x 28 - End with empty line

Pixel Values
  0: no edge (blank)
  1: gray pixel (+) [used for digits only]
  2: edge [for face] or black pixel [for digit] (#)

Digits:
Test: 1000
Training: 5000
Validate: 1000

Face:
Test: 150
Training: 451
Validate: 301
'''