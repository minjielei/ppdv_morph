import numpy as np
import torch
from scattering.utils import to_numpy

# for large data, scattering computation needs to be chunked to hold on memory
def chunk_model2d(Xa,Xb,st_calc,nchunks,**kwargs):
    partition = np.array_split(np.arange(Xa.shape[0]), nchunks)
    corr_arr = []
    c00_summary_arr = [] 
    c11_summary_arr = []
    c00_reduced_arr = []
    c11_reduced_arr = []
    for part in partition:
        Xa_here = Xa[part,:,:]
        Xb_here = Xb[part,:,:]
        
        s_cov_here = st_calc.scattering_cross_corr(Xa_here, Xb_here, normalization='P11')

        corr = s_cov_here['corr'].cpu().numpy()
        c00_summary = s_cov_here['c00_summary'].cpu().numpy()
        c11_summary = s_cov_here['c11_summary'].cpu().numpy()
        c00_reduced = s_cov_here['c00_reduced'].cpu().numpy()
        c11_reduced = s_cov_here['c11_reduced'].cpu().numpy()
        
        corr_arr.append(corr)
        c00_summary_arr.append(c00_summary)
        c11_summary_arr.append(c11_summary)
        c00_reduced_arr.append(c00_reduced)
        c11_reduced_arr.append(c11_reduced)
    s_cov_set = {'corr': np.concatenate(corr_arr, axis=0),
                 'c00_summary': np.concatenate(c00_summary_arr, axis=0),
                 'c11_summary': np.concatenate(c11_summary_arr, axis=0),
                 'c00_reduced': np.concatenate(c00_reduced_arr, axis=0),
                 'c11_reduced': np.concatenate(c11_reduced_arr, axis=0)}
    return s_cov_set

# Compute ST correlation coefficients for 3D ppv or ppd data cubes
def compute_st_corr_cube(data_cube, st_calc, nchunks=1):
    M, N = data_cube.shape[-1], data_cube.shape[-2]
    nchannel = data_cube.shape[0]
    Xa = np.broadcast_to(data_cube[:,np.newaxis,...], (nchannel, *data_cube.shape)).reshape((-1, M, N))
    Xb = np.broadcast_to(data_cube[np.newaxis,...], (nchannel, *data_cube.shape)).reshape((-1, M, N))
    s_cov_set = chunk_model2d(Xa, Xb, st_calc, nchunks)
    return s_cov_set

# Compute ST correlation coefficients between two 3D data cubes
def compute_st_corr_2channel(data_cube1, data_cube2, st_calc, nchunks=1):
    M, N = data_cube1.shape[-1], data_cube1.shape[-2]
    nchannel1, nchannel2 = data_cube1.shape[0], data_cube2.shape[0]
    Xa = np.broadcast_to(data_cube1[:,np.newaxis,...], (nchannel1, nchannel2, *data_cube1.shape[1:])).reshape((-1, M, N))
    Xb = np.broadcast_to(data_cube2[np.newaxis,...], (nchannel1, nchannel2, *data_cube2.shape[1:])).reshape((-1, M, N))
    s_cov_set = chunk_model2d(Xa, Xb, st_calc, nchunks)
    return s_cov_set