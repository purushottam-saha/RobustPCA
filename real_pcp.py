import numpy as np
from utility import thres
import preprocessing
from datetime import datetime
from tqdm import trange

# def power_svd(A, niters=100,tol=1e-6):
#     """Compute SVD using Power Method.
#     Input:
#             A: Input matrix which needs to be compute SVD.
#             iters: # of iterations to recursively compute the SVD.
#     Output:
#             u: Left singular vector of current singular value.
#             sigma: Singular value in current iteration.
#             v: Right singular vector of current singular value.
#     """
#     x = np.random.normal(0,1, size=A.shape[1])
#     B = A.T.dot(A)
#     #print('Done once')
#     for i in range(niters):
#         new_x = B.dot(x)
#         x = new_x
#         if np.linalg.norm(x-new_x)<tol:
#             break
#     v = x / np.linalg.norm(x)
#     sigma = np.linalg.norm(A.dot(v))
#     u = A.dot(v) / sigma
#     return np.reshape(
#         u, (A.shape[0], 1)), sigma, np.reshape(
#         v, (A.shape[1], 1))

def power_svd_max(M,tol=1e-2,niter=100000):
    m, n = M.shape
    v = np.random.normal(0,1, size=n)
    u = np.random.normal(0,1, size=m)
    for i in range(niter):
        u_tmp = u
        v_tmp = v
        u = M.dot(v_tmp)/np.linalg.norm(M.dot(v_tmp))
        v = M.T.dot(u_tmp)/np.linalg.norm(M.T.dot(u_tmp))
        sigma = np.linalg.norm(M.T.dot(u_tmp))
        if np.linalg.norm(v-v_tmp)< tol:
            print("converged at",i)
            break
    else:
        print("not converged")
    return np.reshape(u, (m, 1)), sigma, np.reshape(v, (n, 1))

def power_svd_trans(M,thresh,niter=100,rank_max=2):
    s = []
    m, n = M.shape
    rank_mx = rank_max if rank_max else min(m,n)
    rank = 0
    M_out = np.zeros((m,n))
    for _ in range(rank_mx):
        u, sigma, v = power_svd_max(M)
        s.append(round(sigma,3))
        if sigma<thresh:
            break
        M = M - u.dot(v.T).dot(sigma)
        M_out += u.dot(v.T).dot(sigma-thresh)
        rank+=1
    return s#M_out,rank



# def power_svd_trans(M,thresh,tol=10**(-7),niter=1000,rank_max=None):
#     m, n = M.shape
#     rank = 0
#     sigma = np.inf
#     M_out = np.zeros((m,n))
#     rank_mx = rank_max if rank_max else min(m,n)
#     for _ in range(rank_mx):
#         u,sigma,v = power_svd_max(M,tol=tol,niter=niter)
#         if sigma<thresh:
#             break
#         M = M - sigma*(u.dot(v))
#         M_out += (sigma-thresh)*(u.dot(v.transpose()))
#         rank+=1
#     return M_out,rank

def realpcp(M, lam=np.nan, mu=np.nan, factor=1, tol=10**(-7), maxit=1000):       
    # initialization
    m, n = M.shape
    S = np.zeros((m,n))
    L = np.zeros((m,n))
    Lambda = np.zeros((m,n)) # the dual variable
 
    # parameter setting
    if np.isnan(mu):
        mu = 0.25/np.abs(M).mean()
    if np.isnan(lam):
        lam = 1/np.sqrt(max(m,n)) * float(factor)
        
    # main
    for niter in trange(maxit):
        #normLS = np.linalg.norm(np.concatenate((S,L), axis=1), 'fro')
        X = Lambda / mu + M
        Y = X - S
        dL = L;       
        L,rank = power_svd_trans(Y,1/mu)
        dL = L - dL
        Y = X - L
        dS = S
        S = thres(Y, lam/mu) # softshinkage operator
        dS = S - dS
        Z = M - S - L
        Lambda = Lambda + mu * Z
        
        # stopping criterion
        # if niter%20==0:
        #     RelChg = np.linalg.norm(np.concatenate((dS, dL), axis=1), 'fro') / (normLS + 1)
        #     if RelChg < tol: 
        #         break
    
    return L, S, niter, rank





# A = np.random.normal(5,0.12,size=(100,100))

# t0 = datetime.now()
# print(power_svd_trans(A,1)[:10])
# t1 = datetime.now()
# print(np.linalg.svd(A)[1][:10])
# t2 = datetime.now()

# print(t1-t0,t2-t1)