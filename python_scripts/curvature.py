import os
import glob
from math import sqrt
from numpy import linalg
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from pathlib import Path
import re
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from scipy.spatial import distance
import sys

def unit_vector(x):
    return x/np.linalg.norm(x)


def unit_normal(a, b, c):
    x = np.linalg.det([[1, a[1], a[2]],
                       [1, b[1], b[2]],
                       [1, c[1], c[2]]])
    y = np.linalg.det([[a[0], 1, a[2]],
                       [b[0], 1, b[2]],
                       [c[0], 1, c[2]]])
    z = np.linalg.det([[a[0], a[1], 1],
                       [b[0], b[1], 1],
                       [c[0], c[1], 1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    return (x/magnitude, y/magnitude, z/magnitude)

# area of polygon poly


def poly_area(poly):
    if len(poly) < 3:  # not a plane - no area
        return 0
    total = [0, 0, 0]
    N = len(poly)
    for i in range(N):
        vi1 = poly[i]
        vi2 = poly[(i+1) % N]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = np.dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)




def fit_hypersphere(data, method="Hyper"):
    """returns a hypersphere of the same dimension as the
        collection of input tuples
                (radius, (center))


    """
    num_points = len(data)

    if num_points == 0:
        return (0, None)
    if num_points == 1:
        return (0, data[0])
    dimen = len(data[0])        # dimensionality of hypersphere

    if num_points < dimen+1:
        raise ValueError(
            "Error: fit_hypersphere needs at least {} points to fit {}-dimensional sphere, but only given {}".format(dimen+1, dimen, num_points))

    # central dimen columns of matrix  (data - centroid)
    central = np.matrix(data, dtype=float)      # copy the data
    centroid = np.mean(central, axis=0)
    for row in central:
        row -= centroid

    square_mag = [sum(a*a for a in row.flat) for row in central]
    square_mag = np.matrix(square_mag).transpose()

    if method == "Taubin":

        mean_square = square_mag.mean()
        data_Z = np.bmat(
            [[(square_mag-mean_square)/(2*sqrt(mean_square)), central]])

        u, s, v = linalg.svd(data_Z, full_matrices=False)
        param_vect = v[-1, :]
        # convert from (dimen+1) x 1 matrix to list
        params = [x for x in np.asarray(param_vect)[0]]
        params[0] /= 2*sqrt(mean_square)
        params.append(-mean_square*params[0])
        params = np.array(params)

    else:

        data_Z = np.bmat([[square_mag, central, np.ones((num_points, 1))]])

        u, s, v = linalg.svd(data_Z, full_matrices=False)

        if s[-1]/s[0] < 1e-12:

            param_vect = v[-1, :]
            params = np.asarray(param_vect)[0]

        else:
            Y = v.H*np.diag(s)*v  # v.H gives adjoint
            Y_inv = v.H*np.diag([1./x for x in s])*v

            Ninv = np.asmatrix(np.identity(dimen+2, dtype=float))
            if method == "Hyper":
                Ninv[0, 0] = 0
                Ninv[0, -1] = 0.5
                Ninv[-1, 0] = 0.5
                Ninv[-1, -1] = -2*square_mag.mean()
            elif method == "Pratt":
                Ninv[0, 0] = 0
                Ninv[0, -1] = -0.5
                Ninv[-1, 0] = -0.5
                Ninv[-1, -1] = 0
            else:
                raise ValueError(
                    "Error: unknown method: {} should be 'Hyper', 'Pratt', or 'Taubin'")

            matrix_for_eigen = Y*Ninv*Y

            eigen_vals, eigen_vects = linalg.eigh(matrix_for_eigen)

            positives = [x for x in eigen_vals if x > 0]
            if len(positives)+1 != len(eigen_vals):
                # raise ValueError("Error: for method {} exactly one eigenvalue should be negative: {}".format(method,eigen_vals))
                print("Warning: for method {} exactly one eigenvalue should be negative: {}".format(
                    method, eigen_vals), file=stderr)
            smallest_positive = min(positives)

            A_colvect = eigen_vects[:, list(
                eigen_vals).index(smallest_positive)]

            param_vect = (Y_inv*A_colvect).transpose()

            # convert from (dimen+2) x 1 matrix to array of (dimen+2)
            params = np.asarray(param_vect)[0]



    radius = 0.5 * \
        sqrt(sum(a*a for a in params[1:-1]) - 4 *
             params[0]*params[-1])/abs(params[0])
    center = -0.5*params[1:-1]/params[0]

    center += np.asarray(centroid)[0]

    return (radius, center)

def curvature_from_potential(pdb_file_path, potential_file_path):
    '''
    Given a potential and a surface pdb file, this gives the curvature
    '''

    # reading the pdb file
    pdb_format = PandasPdb()
    pdb_format.read_pdb(pdb_file_path)
    pdb_df = pdb_format.df['ATOM']

    # reading the potential file
    col_specs = [
    (0, 5),
    (5, 8),
    (8,10),
    (10,15),
    (19,31)
    ]
    col_names = ["atom", "res_name", "chain", "res_id", "potential"]
    potential_df = pd.read_fwf(potential_file_path, colspecs=col_specs, header=None, names = col_names,skiprows=12)
    potential_df = potential_df.iloc[:-2]

    # merging the two DataFrames and cleaning it up for further processing
    merged_df = pd.concat([pdb_df, potential_df], axis=1)
    merged_df = merged_df[['atom','res_id','res_name','potential','x_coord','y_coord','z_coord']]
    merged_df.rename(columns={'x_coord':'x','y_coord':'y','z_coord':'z'},inplace=True)
    merged_df['potential'] = merged_df['potential'].astype(np.float64)

    df = merged_df
    cols = ['x','y','z']
    df_no_duplicate = df.drop_duplicates(subset=cols)

    df_sur = df_no_duplicate.copy()
    # the clustering algorithm - heirarchical clustering
    data = df_sur[cols] # choosing only the x y and z coordinates.
    Z = linkage(data, 'complete')
    max_d = 15 # can modify later
    clusters = fcluster(Z, max_d, criterion='distance')
    centroid = np.median(data, axis=0) # finding the median for each of x, y and z
    df_sur['cluster'] = clusters # setting the column clusters to the cluster number.
    header_of_final_df = ['x', 'y', 'z', 'atom', 'res_name', 'res_id','potential','cluster', 'shape', 'curvature']
    df_final = pd.DataFrame(columns=header_of_final_df)

    # code for calculating the curvature:

    for i in range(1, max(clusters)+1):

        # choosing rows of the DataFrame corresponding to a cluster number
        df_of_cluster_orig = df_sur[df_sur['cluster']==i]
        data = df_of_cluster_orig[cols] # choosing the x, y and z coordinates
        df_of_cluster = df_of_cluster_orig.copy() # the cluster's data frame

        # fit hypersphere is a written function in the utils directory.
        curv_m = fit_hypersphere(np.array(data))
        # curv_m[0] - radius and curv_m[1] - center of the hypersphere.
        ci = curv_m[1] # center of the hypersphere
        # computes the distance between the centroid and center of hypersphere.
        d_centroid = np.linalg.norm(centroid-ci)

        for index in df_of_cluster.index:
            # creating a matrix of coordinates
            x = np.array([df_of_cluster['x'][index],df_of_cluster['y'][index],df_of_cluster['z'][index]])
            # distance between center of the fit hypersphere and the point in the cluster
            d = np.linalg.norm(ci-x)
            # distance between median/centroid of all points and the point on the cluster
            d_c = np.linalg.norm(centroid-x)

            # allocating each point as either convex or concave based on the distance calculated
            # 1 - protrusion - positive value of curvature
            # 2 - cavity     - negative value of curvature
            if d_c > d_centroid:
                df_of_cluster.loc[index,'shape'] = 1
            else:
                df_of_cluster.loc[index,'shape'] = 2

        #calculating constants based on number of convex or concave points on a cluster.

        A = (len(df_of_cluster[df_of_cluster['shape'] == 1])/len(df_of_cluster))
        B = (len(df_of_cluster[df_of_cluster['shape'] == 2])/len(df_of_cluster))


        #calculating the curvature of each point based on whether it is concave or convex

        for each_index in (df_of_cluster[df_of_cluster['shape']==1]).index:
            curv_1 = A*100/curv_m[0]
            curvature_1 = round(curv_1,2)
            # appending the curvature in the DataFrame
            df_of_cluster.loc[each_index,'curvature'] = curvature_1
        for each_index in (df_of_cluster[df_of_cluster['shape']==2]).index:
            curv_2 = B*-100/curv_m[0]
            curvature_2 = round(curv_2,2)
            # appending the curvature in the DataFrame
            df_of_cluster.loc[each_index,'curvature'] = curvature_2
        # appending the cluster's DataFrame to the "final DataFrame"
        df_final = pd.concat([df_final,df_of_cluster])
#         df_final.drop(columns =['sur_type','atom_info','closest_sphere','face_number'],inplace=True)
#         saving_file_name = os.path.basename(file).split('.')[0]
#         df_final.to_csv('data/final_data_vert_csv/'+saving_file_name+'.csv')
    return df_final

def all_cluster_info(molecule):
    '''
    Instead of dot wise information, it returns cluster wise information. Input is a DataFrame generated from the charge_molecule     function and output is a DataFrame whose columns are 'cluster', 'curvature',x, ,z and average potential
    '''
    cluster_details = []


    # summing the charge on each cluster and input into a list, similaryl, also find the 'dominant' curvature of the cluster. If curvature value is positive - protrusion, if negative - cavity
    for i in range(1,molecule['cluster'].max()+1):
        # condition = charged_molecule['cluster'] == i
        # sum_b = charged_molecule.loc[condition, 'charge'].sum()
        cluster_wise_df = molecule[molecule['cluster']==i]
        value_counts=cluster_wise_df['curvature'].value_counts()
        max_key = value_counts.idxmax()
        cols = ['x','y','z','potential']
#         centroid = np.median()
        centroid = np.mean(cluster_wise_df[cols],axis=0)
        cluster_details.append([i,max_key,centroid[0],centroid[1],centroid[2],centroid[3]])


    return pd.DataFrame(cluster_details,columns=['cluster_id','curvature','x','y','z','average_potential'])

input_folder_name=sys.argv[1]
output_folder_name=sys.argv[2]
start_number = int(sys.argv[3])
stop_number = int(sys.argv[4])
df_containing_files = pd.read_csv('positive_output_pot_files.csv')
list_of_files = df_containing_files['filename'][start_number:stop_number]

for file_basic_name in list_of_files:
  try:
    potential_file=input_folder_name+'/'+file_basic_name+'.pot'
    surface_file_pdb = input_folder_name+'/'+file_basic_name+'_surf.pdb'
    molecule = curvature_from_potential(pdb_file_path=surface_file_pdb, potential_file_path=potential_file)
    all_cluster_df =all_cluster_info(molecule)
    output_file_path = output_folder_name+'/'+file_basic_name+'.csv'
    all_cluster_df.to_csv(output_file_path)
  except ValueError as e:
    print(str(file_basic_name))
    print("Error: The number of observations cannot be determined on an empty distance matrix.")
  except MemoryError as m:
    print(str(file_basic_name))
    print('Error:memory error')
