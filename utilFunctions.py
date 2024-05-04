import numpy as np

'''
Pixel Values
  0: no edge (blank)
  1: gray pixel (+) [used for digits only]
  2: edge [for face] or black pixel [for digit] (#)
'''

# Converts Ascii input to numerical representation
def Integer_Conversion_Function(character):
  if(character == ' '):
    return 0
  elif(character == '+'):
    return 1
  elif(character == '#'):
    return 2   

# Loads count number of images of size (x_dim, y_dim) from file
# Returns 3-D numpy array (number of image, x_pixel, y_pixel)
def load_Image_Data(file, count, x_dim, y_dim):
  images = np.empty(shape = (count, x_dim, y_dim), dtype=(np.unicode_))

  # Open file. Read each line into numpy array
  with open(file, 'r') as f:
    for i in range(count):
        for j in range(x_dim):
          images[i][j] = [char for char in f.readline()][0:y_dim]

  # Function to convert ascii to integer
  intConvertVector = np.vectorize(Integer_Conversion_Function)

  return intConvertVector(images)

# Loads count number of labels from file
# Return 1-D numpy array of labels
def load_label_Data(file, count):
  labels = np.zeros(shape=count, dtype=(np.int_))

  with open(file, 'r') as f:
    for i in range(count):
        labels[i] = f.readline()

  return labels

# binary activation
def binary_Activation(x):
  if x >= 0: return 1
  else: return 0

# Testing
# np.set_printoptions(threshold=np.inf, linewidth=300)
# print(load_Image_Data("data/facedata/facedatatest", 150, 70, 60)[0])
# print(load_label_Data("data/facedata/facedatatestlabels", 150)[7])