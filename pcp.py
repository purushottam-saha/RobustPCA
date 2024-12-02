import numpy as np
from utility import thres
import preprocessing
from datetime import datetime
from tqdm import trange

def pcp(M, lam=np.nan, mu=np.nan, factor=1, tol=10**(-7), maxit=1000):
    """ Robust PCA (Candes et al., 2009)
    
    This code solves the Principal Component Pursuit
    min_M { ||L||_* + lam*||S(:)||_1 }
    s.t. M = S+L
    using an Augmented Lagrange Multiplier (ALM) algorithm.
  
    Parameters
    ----------
    M : array-like, shape (n_features, n_samples), which will be decomposed into a sparse matrix S 
        and a low-rank matrix L.
        
    lam : positive tuning parameter (default NaN). When lam is set to NaN,  the value 1/sqrt(max(m, n)) * factor 
    will be used in the PCP algorithm, where M is a m by n matrix.
    
    mu: postive tuning parameter used in augmented Lagrangian which is in front of 
    the ||M-L-S||_F^2 term (default NaN). When mu is set to NaN, the value 0.25/np.abs(M).mean() 
    will be used in the PCP algorithm.
    
    factor: tuning parameter (default 1). When lam is set to NaN,  lam will take the value 1/sqrt(max(m, n)) * factor
    in the PCP algorithm, where M is a m by n matrix. When lam is not NaN, factor is ignored.
    
    tol : tolerance value for convergency (default 10^-7).
    
    maxit : maximum iteration (default 1000).
    
    Returns
    ----------
    L : array-like, low-rank matrix.
    
    S : array-like, sparse matrix.
    
    niter : number of iteration.
    
    rank : rank of low-rank matrix.
    
    References
    ----------
    Candes, Emmanuel J., et al. Robust principal component analysis. 
        Journal of the ACM (JACM) 58.3 (2011): 11.
    
    Yuan, Xiaoming, and Junfeng Yang. Sparse and low-rank matrix decomposition via alternating direction methods. 
        preprint 12 (2009). [tuning method]
    
    """            
    # initialization
    m, n = M.shape
    unobserved = np.isnan(M)
    M[unobserved] = 0
    S = np.zeros((m,n))
    L = np.zeros((m,n))
    Lambda = np.zeros((m,n)) # the dual variable
 
    # parameter setting
    if np.isnan(mu):
        mu = 0.25/np.abs(M).mean()
    if np.isnan(lam):
        lam = 1/np.sqrt(max(m,n)) * float(factor)
        
    # main
    print("PCP Running...")
    for niter in trange(maxit):
        normLS = np.linalg.norm(np.concatenate((S,L), axis=1), 'fro')              
        # dS, dL record the change of S and L, only used for stopping criterion

        X = Lambda / mu + M
        # L - subproblem
        # L = argmin_L ||L||_* + <Lambda, M-L-S> + (mu/2) * ||M-L-S||.^2
        # L has closed form solution (singular value thresholding)
        Y = X - S
        dL = L
        U, sigmas, V = np.linalg.svd(Y, full_matrices=False)
        rank = (sigmas > 1/mu).sum()
        Sigma = np.diag(sigmas[0:rank] - 1/mu)
        L = np.dot(np.dot(U[:,0:rank], Sigma), V[0:rank,:])
        dL = L - dL
        
        # S - subproblem 
        # S = argmin_S  lam*||S||_1 + <Lambda, M-L-S> + (mu/2) * ||M-L-S||.^2
        # Define element wise softshinkage operator as 
        #     softshrink(z; gamma) = sign(z).* max(abs(z)-gamma, 0);
        # S has closed form solution: S=softshrink(Lambda/mu + M - L; lam/mu)
        Y = X - L
        dS = S
        S = thres(Y, lam/mu) # softshinkage operator
        dS = S - dS

        # Update Lambda (dual variable)
        Z = M - S - L
        Z[unobserved] = 0
        Lambda = Lambda + mu * Z
        
        # stopping criterion
        RelChg = np.linalg.norm(np.concatenate((dS, dL), axis=1), 'fro') / (normLS + 1)
        #print(f'pcp itertion =  {niter}')
        if RelChg < tol: 
            print("PCP Converged")
            break
    
    return L, S, niter, rank



name = 'fan hand'
algo = 'pcp'
desired_width = 225 #400    16 to 9 ratio
desired_height = 400 #225

if __name__=='__main__':

    start_time = datetime.now()
    
    print("Process started")
    preprocessing.resize_video(f'Data/{name}.mp4', f'Data/resize_{name}.mp4', desired_width, desired_height)
    print("Resizing done")
    preprocessing.to_gif(f'Data/resize_{name}.mp4',f'Data/{name}.gif')
    print("Gif created")
    M,shape,n_frames = preprocessing.get_X_from_gif(f'Data/{name}.gif')
    print(M.shape)
    print(f'Data Loaded, {n_frames} frames and {shape[0]}x{shape[1]} size images.')
    preprocessing.mat_to_gif(M,shape,n_frames,f'output/{name}_bnw.gif')
    print("Main process about to start")
    
    recons, frecons,iteration,rank = pcp(M)


    print(f'complete, rank = {rank}')
    preprocessing.mat_to_gif(recons,shape,n_frames,f'output/{algo}/{name}_bnw_backg.gif')
    preprocessing.mat_to_gif(frecons,shape,n_frames,f'output/{algo}/{name}_bnw_foreg.gif')
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

