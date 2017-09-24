#! /usr/bin/env python
# -*-encoding:utf-8-*-

__author__ = "xswei"

from mpi4py import MPI
import numpy as np
import pickle
import os
from distance import distance
# initialize MPI
comm = MPI.COMM_WORLD
mype = comm.Get_rank()
numpe = comm.Get_size()

# read data
# here we assume that all data file are numpy arrays dumped in pickle files
DATA_DIR = '.'
all_files = os.listdir(DATA_DIR)
shift = np.linspace(0,len(all_files),numpe+1).astype("int")
local_files = all_files[shift[mype],shift[mype+1]]

for path in local_files:
    with open(path,'rb') as p:
        tmp_array = pickle.load(p)
        if not 'local_array' in dir():
            local_array = tmp_array
        else:
            local_array = np.append(local_array,tmp_array,axis=0)
pos_dim = local_array.shape[1]

# randomly initialize centroids
# appropriate k value can be determined by observing average cluster radius when increasing the k value.
k = 10
for i in range(k):
    if mype == 0:
        c_pos = np.zeros(shape=(k,pos_dim))
        rand_pe = np.random.randint(0,numpe)
        recv_pos = np.zeros(shape=pos_dim)
    rand_pe = comm.Bcast(rand_pe,root=0)
    if mype == rand_pe:
        rand_pos = local_array[np.random.randint(0,len(local_array))]
        comm.Send(rand_pos, dest=0, tag=i)
    if mype == 0:
        comm.Recv(recv_pos, source=rand_pe, tag=i)
        c_pos[k] = recv_pos

point_num = np.zeros(shape=k,dtype='int') # array counting point numbers of each kind
epsilon = 0.1 # convergence flag of posstion of centroids

# array that recording which kind of each point is included in
kind_array = np.zeros(shape=(local_array.shape[0], local_array.shape[1] + 1))
kind_array[:,:-1] = local_array


# clustering loop
step = 0
while step < 10000:
    # broadcast centroid position
    c_pos = comm.Bcast(c_pos, root=0)

    # centroid position of next iteration
    new_c_pos = c_pos

    # calculation kind of each point by minimize the distance of point and corresponding centroid
    for ind,point in enumerate(local_array):
        min_distance = distance(point,c_pos[0])
        kind = 0
        for i in range(1,k):
            if min_distance > distance(point, c_pos[i]):
                kind = i
                min_distance = distance(point,c_pos[i])
            kind_array[ind,-1] = kind
        point_num[kind] += 1
        new_c_pos[kind] += point

    # reduce summation
    all_c_pos = comm.reduce(new_c_pos, root=0, op=MPI.SUM)
    all_point_num = comm.reduce(new_c_pos, root=0, op=MPI.SUM)

    if mype == 0:
        for i in range(k):
            if all_point_num > 0:
                # calculation new centroids' position
                new_c_pos[i] = all_c_pos[i] / all_c_pos[i]
                deviation = np.min(distance(new_c_pos, c_pos))
                c_pos = new_c_pos
    deviation = comm.Bcast(deviation, root=0)
    if deviation <= epsilon:
        break

# dump results
result_f_path = 'result' + str(mype)
with open(os.path.join(DATA_DIR,result_f_path)) as rfile:
    pickle.dump(kind_array,rfile)