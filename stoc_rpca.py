# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 10:44:23 2016

@author: wexiao
"""
from pcp import pcp
from utility import solve_proj2
import numpy as np
import preprocessing
from datetime import datetime
from tqdm import trange


def stoc_rpca(M, burnin,  lambda1=np.nan, lambda2=np.nan):
    """ 
    Online Robust PCA via Stochastic Optimizaton (Feng, Xu and Yan, 2013)
 
    The loss function is 
        min_{L,S} { 1/2||M-L-S||_F^2 + lambda1||L||_* + lambda2*||S(:)||_1}
    
    Parameters
    ----------
    M : array-like, shape (n_features, n_samples), which will be decomposed into a sparse matrix S 
        and a low-rank matrix L.
    
    lambda1, lambda2:tuning parameters
    
    burnin : burnin sample size.
    
    Returns
    ----------
    Lhat : array-like, low-rank matrix.
    
    Shat : array-like, sparse matrix.
    
    rank : rank of low-rank matrix.
    
    References
    ----------
    Feng, Jiashi, Huan Xu, and Shuicheng Yan. 
    Online robust pca via stochastic optimization. Advances in Neural Information Processing Systems. 2013.

    Rule of thumb for tuning paramters:
    lambda1 = 1.0/np.sqrt(max(M.shape));
    lambda2 = lambda1;
    
    """
    m, n = M.shape
    # calculate pcp on burnin samples and find rank r
    print('Burnin PCP Starting...')
    Lhat, Shat, niter, r = pcp(M[:,:burnin])
    print("Burnin PCP Complete, rank =",r)
    Uhat, sigmas_hat, Vhat = np.linalg.svd(Lhat,full_matrices=False)
    if np.isnan(lambda1):
        lambda1 = 1.0/np.sqrt(m)/np.mean(sigmas_hat[:r])
    if np.isnan(lambda2):
        lambda2 = 1.0/np.sqrt(m)  
    
    # initialization
    U = Uhat[:,:r].dot(np.sqrt(np.diag(sigmas_hat[:r])))
#    Uhat, sigmas_hat, Vhat = np.linalg.svd(Lhat)
#    U = Uhat[:,:r].dot(np.sqrt(np.diag(sigmas_hat[:r])))
    A = np.zeros((r, r))
    B = np.zeros((m, r))
    print('STOC RPCA Starting...')
    for i in trange(burnin, n):
        mi = M[:, i]
        vi, si = solve_proj2(mi, U, lambda1, lambda2)
        Shat = np.hstack((Shat, si.reshape(m,1)))
        A = A + np.outer(vi, vi)
        B = B + np.outer(mi - si, vi)
        U = update_col(U, A, B, lambda1)
        #Lhat = np.hstack((Lhat, U.dot(vi).reshape(m,1)))
        Lhat = np.hstack((Lhat, (mi - si).reshape(m,1)))
        
    return Lhat, Shat, r, U
    
def update_col(U, A, B, lambda1):
    m, r = U.shape
    A = A + lambda1*np.identity(r)
    for j in range(r):
        bj = B[:,j]
        uj = U[:,j]
        aj = A[:,j]
        temp = (bj - U.dot(aj))/A[j,j] + uj
        U[:,j] = temp/max(np.linalg.norm(temp), 1)
    
    return U



name = 'fan hand'
algo = 'stocrpca'
desired_width = 112 #400    16 to 9 ratio
desired_height = 200 #225

if __name__=='__main__':

    start_time = datetime.now()
    
    print("Process started")
    preprocessing.resize_video(f'mp4vid/{name}.mp4', f'mp4vid/resize_{name}.mp4', desired_width, desired_height)
    print("Resizing done")
    preprocessing.to_gif(f'mp4vid/resize_{name}.mp4',f'gifvid/{name}.gif')
    print("Gif created")
    M,shape,n_frames = preprocessing.get_X_from_gif(f'gifvid/{name}.gif')
    print(M.shape)
    print(f'Data Loaded, {n_frames} frames and {shape[0]}x{shape[1]} size images.')
    preprocessing.mat_to_gif(M,shape,n_frames,f'output/{name}_bnw.gif')
    print("Main process about to start")
    
    recons, frecons,rank, U = stoc_rpca(M.transpose(), burnin = 15)


    print(f'complete, rank = {rank}')
    preprocessing.mat_to_gif(recons.transpose(),shape,n_frames,f'output/{algo}/{name}_bnw_backg.gif')
    preprocessing.mat_to_gif(frecons.transpose(),shape,n_frames,f'output/{algo}/{name}_bnw_foreg.gif')
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

