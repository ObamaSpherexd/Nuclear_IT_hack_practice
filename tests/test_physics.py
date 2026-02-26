'''
Tests to check physics
Startup: python -m pytest tests/test_physics.py -v
'''
import numpy as np
import sys
sys.path.append('src')

from elements import Drift,Quadrupole
from beamline import Beamline
from twiss import make_sigma_from_twiss, get_twiss_from_sigma, check_twiss_identity,get_emittance

def test_drift_matrix():
    '''Checking drift matrix'''
    drift=Drift(1.0)
    M=drift.matrix()
    expected=np.array([[1,1], [0,1]])
    assert np.allclose(M,expected), 'Drift matrix is incorrect'
    print('drift_matrix test passed')

def test_quad_matrix():
    '''Checking quadrupole matrix'''
    quad=Quadrupole(f=2.0)
    M=quad.matrix()
    expected=np.array([[1,0],[-0.5,1]])
    assert np.allclose(M,expected), 'Quadrupole matrix is incorrect'
    print('quad_matrix test passed')

def test_emittance_conservation():
    '''Checks emittance conservation'''
    sigma0=make_sigma_from_twiss(beta=10.0,alpha=0.0,epsilon=1e-6)
    eps0=get_emittance(sigma0)

    bl=Beamline()
    bl.add(Quadrupole(f=5.0)).add(Drift(2.0)).add(Quadrupole(f=-5.0)).add(Drift(2.0))

    sigma_out=bl.track_sigma_to_end(sigma0)
    eps_out=get_emittance(sigma_out)

    assert np.isclose(eps0,eps_out,rtol=1e-10), f'Emittance did not conserve: {eps0} vs {eps_out}'
    print('test_emittance passed')

def test_twiss_identity():
    '''Checking beta*gamma-alpha^2=1'''
    beta,alpha=10.0,0.0
    gamma=(1+alpha**2)/beta
    identity=check_twiss_identity(beta,alpha,gamma)
    assert np.isclose(identity,1.0,rtol=1e-10), f'Equation wrong: {identity}'
    print('test_twiss_identity passed')

def test_stability_condition():
    '''Checking stability condition'''
    bl_stable=Beamline()
    bl_stable.add(Quadrupole(f=5.0)).add(Drift(2.0)).add(Quadrupole(f=-5.0)).add(Drift(2.0))

    stable,trace=bl_stable.is_stable()
    assert stable==True, f'Stable lattice considered unstable (Tr={trace})'
    print('test_stability_condition passed')

if __name__=='__main__':
    test_drift_matrix()
    test_quad_matrix()
    test_emittance_conservation()
    test_twiss_identity()
    test_stability_condition()
    print('all test passed!')