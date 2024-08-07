"""
Created on December 12 22:45:10 2023
@author: Chuan-Shen Hu
"""
##
import numpy as np
from numpy import loadtxt
from numpy import inf
##
import matplotlib.pyplot as plt

## Given a persistence barcode as a list, change the interval (b,inf) to (b, max_filtration_level)
def barcode_normalization(input_PD, max_filtration_level):
  result = []
  for (birth, death) in input_PD:
    if death == inf:
      result.append([birth, max_filtration_level])
    else:
      result.append([birth, death])
  return result

## For any barcode.
def get_d_distribution(input_PD, max_filtration_level, num_grids):
  result = np.zeros((num_grids,))
  new_PD = barcode_normalization(input_PD, max_filtration_level)
  d_values = [death for (birth, death) in new_PD]
  num_barcodes = len(d_values)
  for i in range(num_grids):
    for j in range(num_barcodes):
      if (d_values[j] >= i * max_filtration_level / num_grids) and (d_values[j] < (i + 1) * max_filtration_level / num_grids):
        result[i] = result[i] + 1
  return result / np.sum(result)

## For q-th barcode with q >= 1. 
def get_b_distribution(input_PD, max_filtration_level, num_grids):
  result = np.zeros((num_grids,))
  b_values = [birth for (birth, death) in input_PD]
  num_barcodes = len(b_values)
  for i in range(num_grids):
    for j in range(num_barcodes):
      if (b_values[j] >= i * max_filtration_level / num_grids) and (b_values[j] < (i + 1) * max_filtration_level / num_grids):
        result[i] = result[i] + 1
  return result / np.sum(result)

## For q-th barcode with q >= 1.
def get_life_distribution(input_PD, max_filtration_level, num_grids):
  result = np.zeros((num_grids,))
  new_PD = barcode_normalization(input_PD, max_filtration_level)
  life_values = [death - birth for (birth, death) in new_PD]
  num_barcodes = len(life_values)
  for i in range(num_grids):
    for j in range(num_barcodes):
      if (life_values[j] >= i * max_filtration_level / num_grids) and (life_values[j] < (i + 1) * max_filtration_level / num_grids):
        result[i] = result[i] + 1
  return result / np.sum(result)


