'''Main function for UCI letter and spam datasets.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

from CGAIN_data_loader import data_loader
from CGAIN_CGAIN import cgain
from utils import rmse_loss


def main (args):
  '''Main function for UCI letter and spam datasets.

  Args:
    - data_name: letter or spam
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations

  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  '''

  data_name = args.data_name
  miss_rate = args.miss_rate

  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}

  # Load data and introduce missingness
  ori_data_x, miss_data_x, data_m = data_loader(data_name, miss_rate)

  # Impute missing data
  # imputed_data_x = gain(ori_data_x, miss_data_x, gain_parameters)

  rmse_5fold = cgain(ori_data_x, miss_data_x, gain_parameters)
  # print('5 fold CV RMSE are : ', rmse_5fold )
  return rmse_5fold
  # # Report the RMSE performance
  # rmse = rmse_loss (ori_data_x, imputed_data_x, data_m)
  #
  # print()
  # print('RMSE Performance: ' + str(np.round(rmse, 4)))
  #
  # return imputed_data_x, rmse

if __name__ == '__main__':

    for alpha in [0.1]:#, 0.5, 1]:
    # Inputs for the main function
        parser = argparse.ArgumentParser()
        parser.add_argument(
        '--data_name',
        choices=['letter','spam','default','news', 'breast'],
        default='breast',
        type=str)
        parser.add_argument(
        '--miss_rate',
        help='missing data probability',
        default=0.2,
        type=float)
        parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch',
        default=64,
        type=int)
        parser.add_argument(
        '--hint_rate',
        help='hint probability',
        default=0.9,
        type=float)
        parser.add_argument(
        '--alpha',
        help='hyperparameter',
        default=alpha,
        type=float)
        parser.add_argument(
        '--iterations',
        help='number of training interations',
        default=10000,
        type=int)

        args = parser.parse_args()

        # Calls main function
        # imputed_data, rmse = main(args)
        rmse_10_spam = []
        for i in range(1):#0):
            r = main(args)
            rmse_10_spam.append(r)

        print ('breast dataset - alpha = ', str(alpha))
        print(rmse_10_spam)
        print('Average RMSE over experiments : ', np.average(np.average(rmse_10_spam,axis=1)))
        print('std over experiments : ', np.std(np.average(rmse_10_spam,axis=1)))
        print('_________________________________________________________________________________')
