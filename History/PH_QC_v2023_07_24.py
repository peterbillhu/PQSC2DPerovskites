"""
Created on Wed July 24 22:45:10 2023
@author: Chuan-Shen Hu
"""

## Import packages
import gudhi as gd
import numpy as np
import math
import ripser, persim
from matplotlib import pyplot as plt
from numpy import linalg as LA

############################################################################################
## Basic Gudhi tools
############################################################################################

## Given a set of points, the maximal radius, and the maximal dimension we are concerned about, output the rips complex.
## The output is a list that consists of rips_complex and simplex_tree.
def get_RipsData(set_of_points,length,dimension):
  rips_complex = gd.RipsComplex(points=set_of_points,max_edge_length=length)
  simplex_tree = rips_complex.create_simplex_tree(max_dimension=dimension)
  return [rips_complex,simplex_tree]

## Given a set of points, the maximal radius, and the maximal dimension we are concerned about, output the rips complex.
## The output is a list that consists of rips_complex and simplex_tree.
def get_RipsData_by_distMatrix(distMatrix,length,dimension):
  rips_complex = gd.RipsComplex(distance_matrix=distMatrix,max_edge_length=length)
  simplex_tree = rips_complex.create_simplex_tree(max_dimension=dimension)
  return [rips_complex,simplex_tree]

## Given a set of points, output the alpha complex.
## The output is a list that consists of rips_complex and simplex_tree.
def get_AlphaData(set_of_points):
  alpha_complex = gd.AlphaComplex(set_of_points)
  simplex_tree = alpha_complex.create_simplex_tree()
  return [alpha_complex,simplex_tree]

## Given a complex and a simplex_tree, print the filtration of the simplexes.
def print_complexData(input_RipsData):
  print("The simplex tree=")
  print(input_RipsData[1])
  result_str = 'The simplicial complex is of dimension ' + repr(input_RipsData[1].dimension()) + ' - ' + \
    repr(input_RipsData[1].num_simplices()) + ' simplices - ' + \
    repr(input_RipsData[1].num_vertices()) + ' vertices.'
  print(result_str)
  fmt = '%s -> %.2f'
  for filtered_value in input_RipsData[1].get_filtration():
    print(fmt % tuple(filtered_value))

## Given a simplex_tree, print the filtration of the simplexes.
def print_simplicial_tree(input_st):
  print("The simplex tree=")
  result_str = 'The simplicial complex is of dimension ' + repr(input_st.dimension()) + ' - ' + \
    repr(input_st.num_simplices()) + ' simplices - ' + \
    repr(input_st.num_vertices()) + ' vertices.'
  print(result_str)
  fmt = '%s -> %.2f'
  for filtered_value in input_st.get_filtration():
    print(fmt % tuple(filtered_value))

## Given an input persistence, plot the persistence diagrams.
def plot_persistence_by_gudhi(input_barcodes, dpi=300):
  # Declaration for barcode_list
  barcode_list = []
  # Updates for the barcodes
  for dim in range(len(input_barcodes)):
    for barcode in input_barcodes[dim]:
      barcode_list.append((dim, [barcode[0], barcode[1]]))
  # Plot persistence diagrams
  plt.figure(dpi=300)
  gd.plot_persistence_diagram(barcode_list,fontsize=18)
  plt.show()
  # Plot persistence barcodes
  plt.figure(dpi=300)
  gd.plot_persistence_barcode(barcode_list,fontsize=18)
  plt.show()

## For example, input_barcodes = [barcode_0, barcode_1]
def plot_persistence_diagram(input_barcodes, input_title, dpi=300):
  plt.figure(dpi=300)
  persim.plot_diagrams(input_barcodes, title=input_title)
  plt.show()

## Reduce the barcodes with negative birth/death values.
def reduce_the_barcodes_with_negative_values(input_barcodes):
  number_of_barcodes = len(input_barcodes)
  result = []
  for i in range(number_of_barcodes):
    barcode = input_barcodes[i]
    if (barcode[0] < 0) or (barcode[1] < 0):
      continue
    else:
      result.append(barcode)
  ## Output the result
  return np.array(result)

## Given a simplicial tree, output the persistence barcodes (a list)
def compute_persistence_diagrams(input_st, max_Betti_degree=3, reduce_negative_barcodes=True):
  barcodes = input_st.compute_persistence()
  result = []
  for i in range(max_Betti_degree+1):
    if reduce_negative_barcodes == True:
      current_barcodes = reduce_the_barcodes_with_negative_values(input_st.persistence_intervals_in_dimension(i))
      if (len(current_barcodes) == 0):
        result.append(np.zeros((0,2)))
      else:
        result.append(current_barcodes)
    else:
      current_barcodes = input_st.persistence_intervals_in_dimension(i)
      if (len(current_barcodes) == 0):
        result.append(np.zeros((0,2)))
      else:
        result.append(current_barcodes)
  return result

############################################################################################
# Basic quotient simplicial complex tools
############################################################################################

## Given a simplicial tree and a quotient dictionary, output the point quotient simplicial tree
def get_point_quotient_complex(simplicial_tree, quotient_dictionary):
  # Get the key and value lists of quotient_dictionary
  quotient_dictionary_keys = []
  quotient_dictionary_values = []
  for key in quotient_dictionary:
    quotient_dictionary_keys.append(key)
    quotient_dictionary_values.append(quotient_dictionary[key])
  ## Get zero simplexes
  list_of_vertices = []
  for filtered_value in simplicial_tree.get_filtration():
    if len(tuple(filtered_value)[0]) == 1:
      list_of_vertices.append(tuple(filtered_value)[0][0])
  ## If the length of quotient_dictionary and the number of zero simplexes
  ## are not compatible, then output the original simplicial tree.
  if len(list_of_vertices) != len(quotient_dictionary):
    print('Numbers of the dictionary length and zero simplexes are not comptable...')
    return simplicial_tree
  else:
    ## Add pseudo (i.e., gluing) points and edges
    counter = len(list_of_vertices)
    ## Get the list of quotient classes
    list_of_quotient_classes = list(set(quotient_dictionary_values))
    for class_number in list_of_quotient_classes:
      ##
      if quotient_dictionary_values.count(class_number) == 1:
        continue
      ##
      for vertex in list_of_vertices:
        if quotient_dictionary_keys.count(vertex) == 0:
          continue
        if class_number == quotient_dictionary[vertex]:
          simplicial_tree.insert([counter, vertex],0)
      counter = counter + 1
    ## Gudhi's trick: Extract all gluing 1-loops
    simplicial_tree.insert([counter, counter+1],-2)
    simplicial_tree.insert([counter, counter+2],-2)
    simplicial_tree.insert([counter+1, counter+2],-2)
    simplicial_tree.insert([counter, counter+1, counter+2],-1)
    return simplicial_tree

############################################################################################
# Shifting-type point quotient simplicial complex tools
############################################################################################

## Given two vectors, output the distance of these two vectors
def distance_of_two_points(p1,p2):
  temp = pow(p1[0]-p2[0],2) + pow(p1[1]-p2[1],2) + pow(p1[2]-p2[2],2)
  result = pow(temp,0.5)
  return result

## Given a 3D point-cloud-set and a collection of 3D vectors, translate the point-cloud-set,
## union all the translated sets, and generate the quotient dictionary.
## Both input_point_cloud and translation_vectors are np.arrays of shapes n x 3 and m x 3.
## Also input the distance matrix ('v1', 'v2', ..., etc.).
def get_data_of_translated_points(input_point_cloud, translation_vectors, version='v1'):
  point_types = []
  quotient_dictionary = {}
  result = np.copy(input_point_cloud)
  computation_buff = np.copy(input_point_cloud)
  shape_of_translation_vectors = np.shape(translation_vectors)
  shape_of_input_point_cloud = np.shape(input_point_cloud)
  ## Set the quotient dictionary
  counter = 0
  for i in range(shape_of_input_point_cloud[0]):
    point_types.append(counter)
  for i in range(shape_of_translation_vectors[0]):
    counter = counter + 1
    result = np.concatenate((result, computation_buff + translation_vectors[i]), axis=0)
    for j in range(shape_of_input_point_cloud[0]):
      point_types.append(counter)

  ## Generate the quotient dictionary
  shape_of_all_points = np.shape(result)
  for i in range(shape_of_all_points[0]):
    quotient_dictionary[i] = i % np.shape(input_point_cloud)[0]
  ## Generate the distance matrix
  input_point_cloud_shape = np.shape(input_point_cloud)
  result_shape = np.shape(result)
  # num_of_center_points is the number of points in the input_point_cloud
  num_of_center_points = input_point_cloud_shape[0]
  num_of_all_points = result_shape[0]
  ## Declare the distance matrix:
  dist_matrix = np.ones((num_of_all_points,num_of_all_points))
  if version == 'v1':
    for i in range(num_of_all_points):
      for j in range(i,num_of_all_points):
        ## If the i-th row and j-th row both belong to outside neighbors
        if (i >= num_of_center_points) and (j >= num_of_center_points):
          dist_matrix[i][j] = np.inf
          dist_matrix[j][i] = np.inf
        else:
          dis = distance_of_two_points(result[i],result[j])
          dist_matrix[i][j] = dis
          dist_matrix[j][i] = dis
  elif version == 'v2':
    for i in range(num_of_all_points):
      for j in range(i,num_of_all_points):
        ## If one of the (i,j) rows belongs to the center neighbor.
        if (point_types[i] * point_types[j] == 0) or (point_types[i] == point_types[j]):
          dis = distance_of_two_points(result[i],result[j])
          dist_matrix[i][j] = dis
          dist_matrix[j][i] = dis
        else:
          dist_matrix[i][j] = np.inf
          dist_matrix[j][i] = np.inf
  ## return the result, quotient_dictionary, and distance matrix
  return result, quotient_dictionary, dist_matrix

## The coordinate_system is a 3x3 np.array of the standard basis, the rows are the vectors
## The input extension_range is a pair (list) of integers,
## e.g., [-1,1] corresponds to the coefficients -1, 0, 1 of vectors in the basis.
def get_translations_by_ranges(coordinate_system, extension_range):
  ## Declaration for the translations
  translations = []
  ## Collect the translations
  for j in range(extension_range[0],extension_range[1]+1):
    for k in range(extension_range[0],extension_range[1]+1):
      for l in range(extension_range[0],extension_range[1]+1):
        if j == 0 and k == 0 and l == 0:
          continue
        vector_buff_1 = coordinate_system[0,:]
        vector_buff_2 = coordinate_system[1,:]
        vector_buff_3 = coordinate_system[2,:]
        temp = j * vector_buff_1 + k * vector_buff_2 + l * vector_buff_3
        translations.append(temp)
  ## Output the result
  return np.array(translations)

## The coordinate_system is a 3x3 np.array of the standard basis, the rows are the vectors
## The input shift_coefficients is an nx3 np.array
def get_translations_by_coefficients(coordinate_system, shift_coefficients):
  ## Declaration for the translations
  translations = []
  num_of_coefficient_triples = np.shape(shift_coefficients)[0]
  ## Collect the translations
  for i in range(num_of_coefficient_triples):
    j = shift_coefficients[i,0]
    k = shift_coefficients[i,1]
    l = shift_coefficients[i,2]
    if j == 0 and k == 0 and l == 0:
      continue
    vector_buff_1 = coordinate_system[0,:]
    vector_buff_2 = coordinate_system[1,:]
    vector_buff_3 = coordinate_system[2,:]
    temp = j * vector_buff_1 + k * vector_buff_2 + l * vector_buff_3
    translations.append(temp)
  ## Output the result
  return np.array(translations)

## Get the persistent homology of a point-quotient simplicial complex that is extended by the extension_range.
def get_point_quotient_persistence(input_point_cloud,
                   coordinate_system,
                   extension_range,
                   max_edge_len,
                   max_simplex_dim,
                   version='v1'):
  ## Get the translations, extension points, quotient dictionary, distance matrix, etc.
  translations = get_translations_by_ranges(coordinate_system, extension_range)
  extended_points, quotient_dictionary, dist_matrix = get_data_of_translated_points(input_point_cloud, translations, version=version)
  ## Get the ordinary filtration
  [rips_data, simplicial_tree] = get_RipsData_by_distMatrix(distMatrix=dist_matrix,length=max_edge_len,dimension=max_simplex_dim)
  ## Insert point to implement the point-quotient simplicial complex
  simplex_tree = get_point_quotient_complex(simplicial_tree, quotient_dictionary)
  ## Get the filtraion of point-quotient simplicial complexes
  [barcodes_0, barcodes_1, barcodes_2] = compute_persistence_diagrams(simplex_tree, max_Betti_degree=2, reduce_negative_barcodes=True)
  return barcodes_0, barcodes_1, barcodes_2

## Get the persistent homology of a point-quotient simplicial complex that is extended by the extension_range.
def get_point_quotient_persistence_by_shift_coefficients(input_point_cloud,
                               coordinate_system,
                               shift_coefficients,
                               max_edge_len,
                               max_simplex_dim,
                               version='v1'):
  ## Get the translations, extension points, quotient dictionary, distance matrix, etc.
  translations = get_translations_by_coefficients(coordinate_system, shift_coefficients)
  extended_points, quotient_dictionary, dist_matrix = get_data_of_translated_points(input_point_cloud, translations, version=version)
  ## Get the ordinary filtration
  [rips_data, simplicial_tree] = get_RipsData_by_distMatrix(distMatrix=dist_matrix,length=max_edge_len,dimension=max_simplex_dim)
  ## Insert point to implement the point-quotient simplicial complex
  simplex_tree = get_point_quotient_complex(simplicial_tree, quotient_dictionary)
  ## Get the filtraion of point-quotient simplicial complexes
  [barcodes_0, barcodes_1, barcodes_2] = compute_persistence_diagrams(simplex_tree, max_Betti_degree=2, reduce_negative_barcodes=True)
  return barcodes_0, barcodes_1, barcodes_2

## Given a point cloud and a translation collection, output the PH of the PQSC
def get_general_point_quotient_persistence(input_point_cloud,
                       translations,
                       max_edge_len,
                       max_simplex_dim,
                       version='v1'):
  ## Get the extension points, quotient dictionary, distance matrix, etc.
  extended_points, quotient_dictionary, dist_matrix = get_data_of_translated_points(input_point_cloud, translations, version=version)
  ## Get the ordinary filtration
  [rips_data, simplicial_tree] = get_RipsData_by_distMatrix(distMatrix=dist_matrix,length=max_edge_len,dimension=max_simplex_dim)
  ## Insert point to implement the point-quotient simplicial complex
  simplex_tree = get_point_quotient_complex(simplicial_tree, quotient_dictionary)
  ## Get the filtraion of point-quotient simplicial complexes
  [barcodes_0, barcodes_1, barcodes_2] = compute_persistence_diagrams(simplex_tree, max_Betti_degree=2, reduce_negative_barcodes=True)
  return barcodes_0, barcodes_1, barcodes_2
