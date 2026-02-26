'''
Beam matching module
Contains optimization functions for matching Twiss-parameters
'''

import numpy as np
from scipy.optimize import minimize
from beamline import Beamline
from twiss import make_sigma_from_twiss, get_twiss_from_sigma

def matching_loss(strengths:np.ndarray ,beamline:Beamline ,sigma0:np.ndarray,
                  beta_target:float ,alpha_target:float,epsilon:float=1e-6)->float:
    '''
    strengths: current values of focus lengths of quadrupoles
    Returns: scalar loss function for optimization'''

    # 1. setting parameters into lattice
    beamline.set_quadrupole_strengths(strengths)

    # 2. tracing beam till the end
    sigma_out=beamline.track_sigma_to_end(sigma0)

    # 3. set Twiss-parameters in output
    beta_calc, alpha_calc,_=get_twiss_from_sigma(sigma_out,epsilon)

    # 4. calculating loss function
    loss=(beta_calc-beta_target)**2+(alpha_calc-alpha_target)**2

    return loss


def match_beamline(beamline:Beamline ,sigma0:np.ndarray ,
                   beta_target:float, alpha_target:float,
                   initial_guess:np.ndarray=None,epsilon:float=1e-6):
    '''Automatically selects params of quads for matching'''
    # 1. get initial params
    if initial_guess is None:
        initial_guess=beamline.get_quadrupole_strengths()

    # 2. setting optimization

    result=minimize(
        fun=matching_loss,
        x0=initial_guess,
        args=(beamline,sigma0,beta_target,alpha_target,epsilon),
        method='Nelder-Mead',
        options={'maxiter':1000, 'xatol':1e-8, 'fatol':1e-10}
    )

    # 3. setting optimal parameters
    beamline.set_quadrupole_strengths(result.x)
    
    return {
        'success':result.success,
        'message':result.message,
        'iterations':result.nit,
        'final_loss':result.fun,
        'optimal_strengths':result.x
    }
