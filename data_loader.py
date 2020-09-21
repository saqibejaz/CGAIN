'''Data loader for UCI letter, spam and MNIST datasets.
'''

# Necessary packages
import numpy as np
from utils import binary_sampler
# from keras.datasets import mnist


def data_loader (data_name, miss_rate):
  '''Loads datasets and introduce missingness.
  
  Args:
    - data_name: letter, spam, or mnist
    - miss_rate: the probability of missing components
    
  Returns:
    data_x: original data
    miss_data_x: data with missing values
    data_m: indicator matrix for missing components
  '''
  
  # Load data
  if data_name in ['letter','spam','default','news','breast']:
    file_name = 'data/master/'+data_name+'.csv'
    data_x_y = np.loadtxt(file_name, delimiter=",", skiprows=1)
  # elif data_name == 'mnist':
  #     (data_x, _), _ = mnist.load_data()
  #     data_x = np.reshape(np.asarray(data_x), [60000, 28*28]).astype(float)

  # Parameters
  data_x = data_x_y[:,:-1]
  y = data_x_y[:,-1]

  no, dim = data_x.shape
  y = np.reshape(y, (no, 1))

  # Introduce missing data
  data_m = binary_sampler(1-miss_rate, no, dim)
  miss_data_x = data_x.copy()
  miss_data_x[data_m == 0] = np.nan

  miss_data_x_y = np.append(miss_data_x,y,axis=1)

  data_m = np.append(data_m, np.ones_like(y), axis=1)

  return data_x_y, miss_data_x_y, data_m
