'''
Accelerator elements module (UPDATED VERSION)
Contains classes for magnetic elements: drift, quadrupole, ACCEPTS THICK LENSES AND DIPOLES'''
import numpy as np

class Element:
    '''Basic accelerator element class'''
    def __init__(self, name:str, length:float=0.0):
        self.name = name
        self.length = length
    
    def matrix_x(self) -> np.ndarray:
        '''Returns transfer matrix for X axis (2x2)'''
        raise NotImplementedError
    
    def matrix_y(self) -> np.ndarray:
        '''Returns transfer matrix for Y axis (2x2)'''
        raise NotImplementedError
    
    def matrix_4d(self)->np.ndarray:
        '''Transfer matrix 4D (4x4) for both axes'''
        Mx=self.matrix_x()
        My=self.matrix_y()
        M=np.zeros((4,4))
        M[0:2,0:2]=Mx
        M[2:4, 2:4]=My
        return M
    
    def track_4d(self, state_4d:np.ndarray) ->np.ndarray:
        '''Tracing a 4D state vector'''
        return self.matrix_4d()@state_4d

class Drift(Element):
    '''Drift element (straight area)'''
    def __init__(self, L):
        super().__init__(f"Drift_{L:.2f}", L)
        self.L = L
    
    def matrix_x(self) ->np.ndarray:
        return np.array([[1, self.L], [0, 1]])
    
    def matrix_y(self) ->np.ndarray:
        return np.array([[1, self.L], [0, 1]])

class Quadrupole(Element):
    '''Quadrupole (focusing lens) WITH THICK LENSES
    
    Args:
        k: field gradient [1/m^2] (k>0 focus, k<0 defocus)
        length: quadrupole length
        f: focus length [m] (k alternative for thin lenses)
    '''
    def __init__(self, k:float=None, length:float=0.0, f:float=None, name:str="Quad"):
        super().__init__(name, length)
        self.f=None
        self.k=None

        if k is not None:
            self.k=k
            if length==0:
                self.f=1.0/k if abs(k)>1e-10 else None

        elif f is not None and length>0:
            # convert f to k for thick lens
            self.k=1/(f*length) if abs(f)>1e-10 else 0
            self.f=f
        elif f is not None:
            # thin lense
            self.k=0
            self.f=f
        else:
            raise ValueError('you need to input k or f')
        
        self.length=length if length>0 else 0.0
    
    
    def matrix_x(self)->np.ndarray:
        '''Matrix for X axis'''
        if self.length==0:
            # thin lens
            return np.array([[1,0],[-1/self.f,1]])
        if abs(self.k)<1e-10:
            # close to drift
            return np.array([[1,self.length],[0,1]])
        if self.k>0:
            # focus in X
            sqrt_k=np.sqrt(self.k)
            kl=sqrt_k*self.length
            return np.array([
                [np.cos(kl),np.sin(kl)/sqrt_k],
                [-sqrt_k*np.sin(kl),np.cos(kl)]
            ])
        else:
            # defocus in X
            sqrt_k=np.sqrt(-self.k)
            kl=sqrt_k*self.length
            return np.array([
                [np.cosh(kl),np.sinh(kl)/sqrt_k],
                [sqrt_k*np.sinh(kl),np.cosh(kl)]
            ])
        
    def matrix_y(self) -> np.ndarray:
        '''Matrix for Y axis (opposite k sign)'''
        if self.length == 0:
            # thin lens - opposite sign
            return np.array([[1, 0], [1/self.f, 1]])
        
        if abs(self.k) < 1e-10:
            # close to drift
            return np.array([[1, self.length], [0, 1]])
        
        # k sign is opposite in Y
        if self.k > 0:
            # focus in X -> DEFOCUS in Y (hyperbolic!)
            sqrt_k = np.sqrt(self.k)
            kl = sqrt_k * self.length
            return np.array([
                [np.cosh(kl), np.sinh(kl)/sqrt_k],  # ← cosh/sinh, не cos/sin!
                [sqrt_k*np.sinh(kl), np.cosh(kl)]
            ])
        else:
            # defocus in X -> FOCUS in Y (trigonometric!)
            sqrt_k = np.sqrt(-self.k)
            kl = sqrt_k * self.length
            return np.array([
                [np.cos(kl), np.sin(kl)/sqrt_k],  # ← cos/sin, не cosh/sinh!
                [-sqrt_k*np.sin(kl), np.cos(kl)]
            ])
        
class Dipole(Element):
    '''
    Dipole (turn magnet)
    
    Args:
        pho: curve radius [m]
        angle: turn angle [rad]
        length: length of a dipole [m] (if none then rho*angle)
        '''
    def __init__(self,rho:float,angle:float,length=None, name='Dipole'):
        if length is None:
            length=rho*angle
        
        super().__init__(name,length)
        self.rho=rho
        self.angle=angle

    def matrix_x(self)->np.ndarray:
        '''Horisontal axis (with focus)'''
        theta=self.angle
        rho=self.rho

        return np.array([
            [np.cos(theta),rho*np.sin(theta)],
            [-np.sin(theta)/rho,np.cos(theta)]
        ])
    
    def matrix_y(self)->np.ndarray:
        '''Vertical axis (as drift)'''
        return np.array([[1,self.length],[0,1]])