'''
Accelerator elements module
Contains classes for magnetic elements: drift, quadrupole'''
import numpy as np

class Element:
    '''Basic accelerator element class'''
    def __init__(self, name:str, length:float=0.0):
        self.name = name
        self.length = length
    
    def matrix(self) -> np.ndarray:
        '''Returns transfer matrix 2x2'''
        raise NotImplementedError
    
    def track_sigma(self, sigma_in:np.ndarray) ->np.ndarray:
        '''Propogation of an covariance matrix through an element'''
        M = self.matrix()
        return M @ sigma_in @ M.T

class Drift(Element):
    '''Drift element (straight area)'''
    def __init__(self, L):
        super().__init__(f"Drift_{L:.2f}", L)
        self.L = L
    
    def matrix(self) ->np.ndarray:
        return np.array([[1, self.L], [0, 1]])

class Quadrupole(Element):
    '''Quadrupole (focusing lens)'''
    def __init__(self, f:float, length:float=0.0, name:str="Quad"):
        super().__init__(name, length)
        self.f = f
    
    def matrix(self)->np.ndarray:
        '''Thin lens matrix'''
        if abs(self.f)<1e-10:
            raise ValueError('Focus length cannot be zero')
        return np.array([[1, 0], [-1/self.f, 1]])
