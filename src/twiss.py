'''
Twiss parameters module
Contains functions for working with alpha, beta, gamma and emittance
'''

import numpy as np

def make_sigma_from_twiss(beta:float, alpha:float,epsilon:float)->np.ndarray:
    '''Creates a covairiance matrix from Twiss parameters'''
    gamma0 = (1 + alpha**2) / beta
    return epsilon * np.array([[beta, -alpha], [-alpha, gamma0]])

    
def get_twiss_from_sigma(sigma:np.ndarray,epsilon:float)->tuple[float,float,float]:
    '''Gets Twiss-parameters from Sigma-matrix'''
    beta=sigma[0,0]/epsilon
    alpha=-sigma[0,1]/epsilon
    gamma=sigma[1,1]/epsilon
    return beta,alpha,gamma

def check_twiss_identity(beta:float,alpha:float,gamma:float)->float:
    '''Checks equation beta*gamma-alpha^2=1'''
    return beta*gamma-alpha**2

def get_emittance(sigma:np.ndarray)->float:
    '''Calculates emittance as sqrt(det(Sigma))'''
    return np.sqrt(np.linalg.det(sigma))