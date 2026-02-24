import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds

# classes

# PARENT 
class Element:
    def __init__(self,name,length=0.0): # is automatically called when creating an object, initial attributes
        self.name=name # self - link to itself
        self.length=length
        
    def matrix(self):
        raise NotImplementedError()
        
    def track_sigma(self,sigma_in):
        '''пропогация матрицы ковариации: Sigma_out=M * Sigma_in * M^T'''
        M=self.matrix()
        return M @ sigma_in @ M.T
    
class Drift(Element):
    def __init__(self,L):
        super().__init__(f'Drift_{L:.2f}',L) # inherits params from parent
        self.L=L

    def matrix(self):
        return np.array([[1,self.L],
                         [0,1]])
    
class Quadrupole(Element):
    def __init__(self, f,length=0.0,name="Quad"):
        super().__init__(name,length)
        self.f=f
        
    def matrix(self):
        return np.array([[1,0],
                         [-1/self.f,1]])

class Beamline:
    def __init__(self):
        self.elements=[]
    
    def add(self, element):
        self.elements.append(element)
        return self
    

    
    def track_sigma_to_end(self, sigma0):
        '''
        Tracing Sigma-matrix till the end
        '''
        sigma=sigma0.copy()
        
        for elem in self.elements:
            sigma=elem.track_sigma(sigma)

                
        return sigma
    
    def get_twiss_from_sigma(self,sigma,epsilon=1e-6):
        '''Gets Twiss-parameters from Sigma-matrix'''
        beta=sigma[0,0]/epsilon
        alpha=-sigma[0,1]/epsilon
        gamma=sigma[1,1]/epsilon
        return beta,alpha,gamma
    
    def set_quadrupole_strengths(self,strengths):
        '''Sets quadrupoles strengths (f=strengths[i])'''
        quad_index=0
        for elem in self.elements:
            if isinstance(elem, Quadrupole):
                if quad_index<len(strengths):
                    elem.f=strengths[quad_index]
                quad_index+=1
    
    def get_quadrupole_strengths(self):
        '''Returns current quads forces'''
        return [elem.f for elem in self.elements if isinstance(elem,Quadrupole)]
        
# LOSS FUNCTION
def matching_loss(strengths,beamline,sigma0,beta_target,alpha_target,epsilon=1e-6):
    '''
    strengths: current values of focus lengths of quadrupoles
    Returns: scalar loss function'''

    # 1. setting parameters into lattice
    beamline.set_quadrupole_strengths(strengths)

    # 2. tracing beam till the end
    sigma_out=beamline.track_sigma_to_end(sigma0)

    # 3. set Twiss-parameters in output
    beta_calc, alpha_calc,_=beamline.get_twiss_from_sigma(sigma_out,epsilon)

    # 4. calculating loss function
    loss=(beta_calc-beta_target)**2+(alpha_calc-alpha_target)**2

    return loss

# MATCHING FUNCTION
def match_beamline(beamline,sigma0,beta_target,alpha_target,
                   initial_guess=None,epsilon=1e-6):
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
    
    return result

# creating initial Sigma-matrix
def make_sigma_from_twiss(beta0 ,alpha0, epsilon):
    '''Creates a covariance matrix for Twiss parameters'''
    gamma0=(1+alpha0**2)/beta0
    sigma=epsilon*np.array([[beta0,-alpha0],
                            [-alpha0, gamma0]])
    return sigma


# PARAMETERS
EPSILON=1E-6
BETA_IN=10.0 # input beta
ALPHA_IN=0.0 # input alpha

# creating entrance beam
sigma0=make_sigma_from_twiss(BETA_IN,ALPHA_IN,EPSILON)

# creating lattice for matching
# entrance - Q1 - drift - Q2 - drift - - Q3 - exit
bl=Beamline()
bl.add(Quadrupole(f=5.0, name='Q1'))
bl.add(Drift(1.5))
bl.add(Quadrupole(f=5.0,name='Q2'))
bl.add(Drift(1.5))
bl.add(Quadrupole(f=5.0,name='Q3'))
bl.add(Drift(1.5))

# target values on exit
BETA_TARGET=5.0 # want to shrink beam
ALPHA_TARGET=0.0 # at waist

print('BEFORE OPTIMIZATION')
initial_strengths=bl.get_quadrupole_strengths()
print(f'Starting f: {initial_strengths}')

sigma_before=bl.track_sigma_to_end(sigma0)
beta_before,alpha_before,_=bl.get_twiss_from_sigma(sigma_before,EPSILON)
print(f'beta on exit: {beta_before:.4f} m (target: {BETA_TARGET})')
print(f'alpha on exit: {alpha_before:.4f} (target: {ALPHA_TARGET})')

# INITIALIZING OPTIMIZATION
print(f'\nOptimization start')
result=match_beamline(bl,sigma0,BETA_TARGET,ALPHA_TARGET,
                      initial_guess=initial_strengths,epsilon=EPSILON)

print(f'Status: {result.message}')
print(f'Iterations: {result.nit}')
print(f'Loss function: {result.fun:.2e}')

# CHECKING THE RESULT
print('\nAFTER OPTIMIZATION')
final_strengths=bl.get_quadrupole_strengths()
print(f"Optimal f' {[f'{f:.4f}' for f in final_strengths]}")

sigma_after=bl.track_sigma_to_end(sigma0)
beta_after,alpha_after,_=bl.get_twiss_from_sigma(sigma_after,EPSILON)
print(f'beta on exit: {beta_after:.4f} (target: {BETA_TARGET})')
print(f'alpha on exit: {alpha_after:.4f} (target: {ALPHA_TARGET})')

print(f"Error: {np.sqrt((beta_after-BETA_TARGET)**2 + (alpha_after-ALPHA_TARGET)**2):.2e}")
# error should be <1e-3

# VISUALIZATION
plt.figure(figsize=(12, 5))

# GRAPH 1: beta before and after
s_before = [0]
beta_before_hist = [BETA_IN]
s_pos = 0
bl_test = Beamline()
bl_test.add(Quadrupole(f=initial_strengths[0]))
bl_test.add(Drift(1.5))
bl_test.add(Quadrupole(f=initial_strengths[1]))
bl_test.add(Drift(1.5))
sigma_test = sigma0.copy()
for elem in bl_test.elements:
    sigma_test = elem.track_sigma(sigma_test)
    s_pos += elem.length
    beta, _, _ = bl_test.get_twiss_from_sigma(sigma_test, EPSILON)
    s_before.append(s_pos)
    beta_before_hist.append(beta)

s_after = [0]
beta_after_hist = [BETA_IN]
s_pos = 0
sigma_test = sigma0.copy()
for elem in bl.elements:
    sigma_test = elem.track_sigma(sigma_test)
    s_pos += elem.length
    beta, _, _ = bl.get_twiss_from_sigma(sigma_test, EPSILON)
    s_after.append(s_pos)
    beta_after_hist.append(beta)

plt.subplot(1, 2, 1)
plt.plot(s_before, beta_before_hist, 'r--', label='before matching', alpha=0.7)
plt.plot(s_after, beta_after_hist, 'b-', label='after matching', linewidth=2)
plt.axhline(BETA_TARGET, color='green', linestyle=':', label=f'Цель β={BETA_TARGET}')
plt.xlabel('travel s [м]')
plt.ylabel('β [м]')
plt.title('beta before and after matching')
plt.grid(True, alpha=0.3)
plt.legend()

# graph 2: matching stability
plt.subplot(1, 2, 2)
plt.plot(result.nit, result.fun, 'go', markersize=10, label='Final')
plt.xlabel('iteration N')
plt.ylabel('loss function')
plt.title('Optimization stability')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('day5_matching.png', dpi=150)
plt.show()


# FUNNIES
'''bounds=Bounds([0.5,0.5],[20.0,20.0])

result=minimize(
    fun=matching_loss,
    x0=initial_guess,
    args=(bl,sigma0,BETA_TARGET,ALPHA_TARGET,EPSILON),
    method='L-DFGS-B', # method accepts bounds
    bounds=bounds,
    options={'maxiter':1000}
)
'''