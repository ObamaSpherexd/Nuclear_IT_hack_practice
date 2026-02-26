'''
Quick demo for demonstration
Startup: python run_demo.py
'''

import sys
import os

current_dir=os.path.dirname(os.path.abspath(__file__))
src_path=os.path.join(current_dir,'src')
sys.path.insert(0,src_path)


import numpy as np
import matplotlib.pyplot as plt

from elements import Drift,Quadrupole
from beamline import Beamline
from twiss import make_sigma_from_twiss, get_twiss_from_sigma
from matching import match_beamline


def main():
    print('Beam Accelerator Simulation - Demo')

    # PARAMETERS
    EPSILON=1E-6
    BETA0=10.0
    ALPHA0=0.0
    sigma0=make_sigma_from_twiss(BETA0,ALPHA0,EPSILON)

    # LATTICE
    bl=Beamline()
    bl.add(Quadrupole(f=5.0,name='QF')).add(Drift(2.0)).add(Quadrupole(f=-5.0,name='QD')).add(Drift(2.0))

    # checking stability
    stable,trace=bl.is_stable()
    print(f'Stability: {'hell yea' if stable else 'hell nah'} (Tr={trace:.4f})')

    # tracing
    s, beta=bl.get_beta_along(sigma0,EPSILON)
    print(f'beta-function: min={min(beta):.2f} m, max={max(beta):2f} m')

    # matching
    print('\nMatching start')
    result=match_beamline(bl,sigma0,beta_target=5.0,alpha_target=0.0,epsilon=EPSILON)
    print(f'Matching: {result['message']} (iterations: {result['iterations']})')

    # graph
    
    s_after, beta_after=bl.get_beta_along(sigma0,EPSILON)

    plt.figure(figsize=(10,5))
    plt.plot(s,beta,'r--',alpha=0.5,label='Before Matching')
    plt.plot(s_after,beta_after,'b-',linewidth=2,label='After Matching')
    plt.xlabel('s [m]')
    plt.ylabel('beta [m]')
    plt.title('Beam Coordination')
    plt.grid(True,alpha=0.3)
    plt.legend()
    plt.savefig('demo_output.png',dpi=150)
    print('\nGraph saved: demo_output.png')
    plt.show()
if __name__=='__main__':
    main()
