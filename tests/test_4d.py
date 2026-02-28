'''
Tests for 4D tracing
'''
import numpy as np
import sys
sys.path.append('src')

from elements import Drift, Quadrupole, Dipole
from beamline import Beamline

def test_quadrupole_4d():
    '''Test: quad focuses in X, defocuses in Y'''
    quad=Quadrupole(k=0.5,length=0.5,name='QF')

    # starting state: displaced on x and y
    state_4d=np.array([0.01,0,0.01,0]) # [x,x',y,y']

    state_out=quad.track_4d(state_4d)

    print(f'Start x= {state_4d[0]:.4f}, y= {state_4d[2]:.4f}')
    print(f'Exit x= {state_out[0]:.4f}, y= {state_out[2]:.4f}')
    print(f'Angle x: {state_out[1]:.4f}, angle y= {state_out[3]:.4f}')

    # X angle has to become negative (focus)
    assert state_out[1]<0, 'Quad has to focus on X'

    # Y angle has to become positive (defocus)
    assert state_out[3]>0, 'Quad has to defocus on Y'

    print('quadrupole_test_4d passed')

def test_dipole_bending():
    '''Test: dipole bends the beam'''
    rho=10.0 # radius
    angle=np.pi/4 # 45 deg
    dipole=Dipole(rho,angle,name='Bend')

    # particle on ideal orbit
    state_4d=np.array([0,0,0,0])
    state_out=dipole.track_4d(state_4d)

    print(f'Dipole length: {dipole.length:.4f} m')
    print(f'Expected length: {rho*angle:.4f} m ')

    assert np.isclose(dipole.length,rho*angle), 'dipole length is incorrect'

    print('test_dipole_bending passed')

def test_fodo_4d_stability():
    '''Tests 4D FODO lattice stability'''
    bl=Beamline()
    bl.add(Quadrupole(k=0.1,length=0.5,name='Qf')).add(Drift(2.0)).add(Quadrupole(k=-0.1,length=0.5,name='QD')).add(Drift(2.0))

    stable,trace_x, trace_y=bl.is_stable_4d()
    print(f'Stability: {stable}')
    print(f'Tr(X)= {trace_x:.4f}, Tr(Y)= {trace_y:.4f}')

    assert stable == True, 'lattice has be be stable'
    print('test_fodo_4d_stability passed')

if __name__=='__main__':
    test_quadrupole_4d()
    test_dipole_bending()
    test_fodo_4d_stability()
    print('ALL TESTS PASSED')
