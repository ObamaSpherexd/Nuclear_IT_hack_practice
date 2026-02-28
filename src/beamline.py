'''
Accelerator lattice module
Contains Beamline class for managing a set of elements
(UPDATED FOR 4D)'''

import numpy as np
from typing import List,Tuple
from elements import Element, Quadrupole, Sextupole


class Beamline:
    '''Set of accelerator particles'''
    def __init__(self):
        self.elements: List[Element]=[]
    
    def add(self, element:Element)->'Beamline':
        '''Adds and element to the lattice'''
        self.elements.append(element)
        return self
    
    def one_turn_matrix_4d(self)->np.ndarray:
        '''Calculates a full lap matrix 4D'''
        M=np.eye(4)
        for elem in self.elements:
            M=elem.matrix_4d()@M
        return M
    
    def is_stable_4d(self)->Tuple[bool,float,float]:
        '''Checks stability condition |Tr(M)|<2 in both axis
        Returns: (stable,trace_x,trace_y)
        '''
        M=self.one_turn_matrix_4d()
        Mx=M[0:2,0:2]
        My=M[2:4,2:4]
        trace_x=np.trace(Mx)
        trace_y=np.trace(My)
        stable=(abs(trace_x)<2) and (abs(trace_y)<2)
        
        return stable,trace_x,trace_y
    
    def track_4d(self,state_4d:np.ndarray) ->np.ndarray:
        '''Tracing 4D state vector to the end of the lattice'''
        state=state_4d.copy()
        for elem in self.elements:
            state=elem.track_4d(state)
        return state
    
    def get_length(self)->float:
        '''Returns the full lattice length'''
        return sum(elem.length for elem in self.elements)
    
    # Old methods for back compatibility
    def one_turn_matrix(self)->np.ndarray:
        '''2D-matrix (only X) for back compatibility'''
        M=np.eye(2)
        for elem in self.elements:
            M=elem.matrix_x() @ M
        return M
    
    def is_stable(self)->Tuple[bool,float]:
        '''2D stability check'''
        stable,trace_x,_=self.is_stable_4d()
        return stable,trace_x

    # === Sigma tracking (for Twiss) ===
    def track_sigma_to_end(self, sigma0: np.ndarray) -> np.ndarray:
        '''Tracing Sigma matrix to the end of the lattice'''
        sigma = sigma0.copy()
        for elem in self.elements:
            # using matrix_x for 2D tracing
            M = elem.matrix_x()
            sigma = M @ sigma @ M.T
        return sigma
    
    def get_beta_along(self, sigma0: np.ndarray, epsilon: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        '''Calculates beta along the whole lattice'''
        sigma = sigma0.copy()
        s_positions = [0]
        beta_history = [sigma[0, 0] / epsilon]
        
        for elem in self.elements:
            M = elem.matrix_x()
            sigma = M @ sigma @ M.T
            s_positions.append(s_positions[-1] + elem.length)
            beta_history.append(sigma[0, 0] / epsilon)
        
        return np.array(s_positions), np.array(beta_history)
    
    def set_quadrupole_strengths(self, strengths: List[float]):
        '''Sets quadrupole strengths (k values for thick lenses)'''
        quad_index = 0
        for elem in self.elements:
            if isinstance(elem, Quadrupole):
                if quad_index < len(strengths):
                    # using k for thick
                    if elem.length > 0:
                        elem.k = strengths[quad_index]
                    else:
                        # using f for thin
                        elem.f_thin = strengths[quad_index]
                quad_index += 1
    
    def get_quadrupole_strengths(self) -> List[float]:
        '''Returns current quadrupole strengths'''
        strengths = []
        for elem in self.elements:
            if isinstance(elem, Quadrupole):
                if elem.length > 0:
                    strengths.append(elem.k)
                else:
                    strengths.append(elem.f_thin)
        return strengths
    
    def track_particle_nonlinear(self,state_4d:np.ndarray)->np.ndarray:
        '''
        Tracing one particle through the whole lattice remembering nonlinearity
        
        Args:
            state_4d: initial state: [x,x',y,y']
            
        Returns:
            End state after lattice
        '''
        state=state_4d
        for elem in self.elements:
            if isinstance(elem,Sextupole):
                # nonlinear element - use kick
                state=elem.track_particle_nonlinear(state)
            else:
                # linear element - use matrix
                state=elem.track_4d(state)

        return state
    
    def track_beam_nonlinear(self,particles:np.ndarray)->np.ndarray:
        '''
        Tracing multiple particles through a nonlinear lattice
        
        Args:
            particles: array (N,4) - N particles, each [x,x',y,y']
        
        Returns:
            array (N,4) - end states
        '''
        output=particles.copy()

        for i in range(len(particles)):
            output[i]=self.track_particle_nonlinear(particles[i])
        return output
    
    def get_dynamic_aperture(self,n_particles:int=1000,
                             max_amplitude:float=0.01)->Tuple[np.ndarray,np.ndarray]:
        '''Evaluates the dynamic aperture (area of stability)
        
        Returns:
            (amplitudes, stable_mask) - amplitudes and stability flag
        '''
        amplitudes=np.linspace(0.001,max_amplitude,n_particles)
        stable_mask=np.zeros(n_particles,dtype=bool)

        for i, amp in enumerate(amplitudes):
            # test particle with given amplitude
            state=np.array([amp,0,0,0]) # only x moved

            # trace for 100 laps
            stable=True
            for turn in range(100):
                state=self.track_particle_nonlinear(state)

                # check whether the particle flew away
                if abs(state[0])>0.1 or abs(state[2])>0.1: # aperture is 10 sm
                    stable=False
                    break
            stable_mask[i]=stable
        return amplitudes,stable_mask
    