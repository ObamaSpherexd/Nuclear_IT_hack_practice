'''
Accelerator lattice module
Contains Beamline class for managing a set of elements'''

import numpy as np
from typing import List,Tuple
from elements import Element, Quadrupole


class Beamline:
    '''Set of accelerator particles'''
    def __init__(self):
        self.elements: List[Element]=[]
    
    def add(self, element:Element)->'Beamline':
        '''Adds and element to the lattice'''
        self.elements.append(element)
        return self
    
    def one_turn_matrix(self)->np.ndarray:
        '''Calculates a full lap matrix'''
        M=np.eye(2)
        for elem in self.elements:
            M=elem.matrix()@M
        return M
    
    def is_stable(self)->Tuple[bool,float]:
        '''Checks stability condition |Tr(M)|<2'''
        M=self.one_turn_matrix()
        trace=np.trace(M)
        return abs(trace)<2, trace
    
    def track_sigma_to_end(self,sigma0:np.ndarray) ->np.ndarray:
        '''Tracing Sigma matrix to the end of the lattice'''
        sigma=sigma0.copy()
        for elem in self.elements:
            sigma=elem.track_sigma(sigma)
        return sigma
    
    def get_beta_along(self,sigma0:np.ndarray,epsilon:float=1e-6)->Tuple[np.ndarray, np.ndarray]:
        '''Calculates beta along the whole lattice'''
        sigma=sigma0.copy()
        s_positions=[0]
        beta_history=[sigma[0,0]/epsilon]

        for elem in self.elements:
            sigma=elem.track_sigma(sigma)
            s_positions.append(s_positions[-1]+elem.length)
            beta_history.append(sigma[0,0]/epsilon)
        return np.array(s_positions),np.array(beta_history)
    
    def set_quadrupole_strengths(self,strengths:List[float]):
        '''Sets focus lengths for quadrupoles'''
        quad_index=0
        for elem in self.elements:
            if isinstance(elem,Quadrupole):
                if quad_index<len(strengths):
                    elem.f=strengths[quad_index]
                quad_index+=1
    def get_quadrupole_strengths(self)->List[float]:
        '''Returns current quadrupole focus lengths'''
        return [elem.f for elem in self.elements if isinstance(elem,Quadrupole)]
    