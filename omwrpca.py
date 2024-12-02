# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 10:44:23 2016

@author: wexiao
"""
from pcp import pcp
from utility import solve_proj2
import numpy as np
from sklearn.utils.extmath import randomized_svd
import preprocessing
from datetime import datetime
from tqdm import trange


def omwrpca(M, burnin, win_size, lambda1=np.nan, lambda2=np.nan, factor=1):
    """ 
    Online Moving Window Robust PCA
    
    The loss function is 
        min_{L,S} { ||L||_* + lam*||S(:)||_1 + 1/{2*mu}||M-L-S||_F^2}
        
    Parameters
    ----------
    M : array-like, shape (n_features, n_samples), which will be decomposed into a sparse matrix S 
        and a low-rank matrix L.
    
    burnin : burnin sample size.
    
    win_size : length of moving window. We require win_size <= burnin.

    lambda1, lambda2:tuning parameters of online moving windows robust PCA.
    
    mu: postive tuning parameter (default NaN). When mu is set to NaN, the value sqrt(2*max(m, n)) 
    will be used in the algorithm, where M is a m by n matrix. A good choice of mu is sqrt(2*max(m, n))*sigma,
    where sigma is the standard deviation of error term.
    
    Returns
    ----------
    Lhat : array-like, low-rank matrix.
    
    Shat : array-like, sparse matrix.
    
    rank : rank of low-rank matrix.
    
    References
    ----------

    Rule of thumb for tuning paramters:
    ----------
    lambda1 = 1.0/np.sqrt(m);
    lambda2 = 1.0/np.sqrt(m);
    
    """
    m, n = M.shape
    # parameter setting
    assert burnin >= win_size, "Parameter burin should be larger than or equal to parameter win_size."
    assert n >= burnin, "Parameter burin should be less than or equal to the number of columns of input matrix."
    if np.isnan(lambda1):
        lambda1 = 1.0/np.sqrt(m)
    if np.isnan(lambda2):
        lambda2 = 1.0/np.sqrt(m)
    # calculate pcp on burnin samples and find rank r
    print("Burnin PCP Starting...")
    Lhat, Shat, niter, r = pcp(M[:, :burnin], factor=factor,maxit=100)
    print("Burnin PCP Complete, rank =",r)
    # initialization for omwrpca
    Uhat, sigmas_hat, Vhat = np.linalg.svd(Lhat, full_matrices=False)
    U = Uhat[:,:r].dot(np.sqrt(np.diag(sigmas_hat[:r])))
    Vhat_win = Vhat[:r, -win_size:]
    A = np.zeros((r, r))
    B = np.zeros((m, r))
    for i in range(Vhat_win.shape[1]):
        A = A + np.outer(Vhat_win[:, i], Vhat_win[:, i])
        B = B + np.outer(M[:, burnin - win_size + i] - Shat[:, burnin - win_size + i], Vhat_win[:, i])
        #print(M[:, burnin - win_size + i].shape)
    
    # main loop
    print("Main OMWRPCA Starting...")
    for i in trange(burnin, n):
        #print(f'i val = {i}')
        mi = M[:, i]
        vi, si = solve_proj2(mi, U, lambda1, lambda2)
        Shat = np.hstack((Shat, si.reshape(m,1)))
        vi_delete = Vhat_win[:,0]
        Vhat_win = np.hstack((Vhat_win[:,1:], vi.reshape(r,1)))
        A = A + np.outer(vi, vi) - np.outer(vi_delete, vi_delete)
        B = B + np.outer(mi - si, vi) - np.outer(M[:, i - win_size] - Shat[:, i - win_size], vi_delete)
        U = update_col(U, A, B, lambda1)
        Lhat = np.hstack((Lhat, (mi - si).reshape(m,1)))
    
    return Lhat, Shat, r
    
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
algo = 'omwrpca'
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
    
    recons, frecons, rank = omwrpca(M.transpose(), burnin = 15, win_size = 10 )


    print(f'complete, rank = {rank}')
    preprocessing.mat_to_gif(recons.transpose(),shape,n_frames,f'output/{algo}/{name}_bnw_backg.gif')
    preprocessing.mat_to_gif(frecons.transpose(),shape,n_frames,f'output/{algo}/{name}_bnw_foreg.gif')
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

