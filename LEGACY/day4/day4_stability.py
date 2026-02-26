import numpy as np
import matplotlib.pyplot as plt

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
    
    def one_turn_matrix(self):
        '''Calculates a full oscillation matrix'''
        M=np.eye(2)
        for elem in self.elements:
            M=elem.matrix() @ M # multiplying on the left
        return M
    
    def is_stable(self):
        '''Checks stability condition'''
        M=self.one_turn_matrix()
        trace=np.trace(M)
        return abs(trace)<2,trace
    
    def get_periodic_twiss(self):
        '''Finds periodic Twiss parameters'''
        M=self.one_turn_matrix()
        trace=np.trace(M)
        if abs(trace)>=2:
            raise ValueError (f'Lattice is unstable! Tr(M)={trace:.4f}')
        mu=np.arccos(trace/2) # phase on 1 oscilation
        sin_mu=np.sin(mu)

        # periodic Twiss parameters

        beta=M[0,1]/sin_mu
        alpha=(M[0,0]-M[1,1])/(2*sin_mu)
        gamma=-M[1,0]/sin_mu

        # Tune (betatrone number)

        tune=mu/(2*np.pi)

        return beta, alpha,gamma, tune
    
    def track_sigma_along(self, sigma0, n_turns=1):
        '''
        Tracing Sigma-matrix for N oscilations
        '''
        sigma=sigma0.copy()
        s_pos=0
        cell_length=sum(e.length for e in self.elements)

        
        # for statistics
        s_history=[0]
        beta_history=[sigma[0,0]]
        turn_history=[0]
        
        

        
        for turn in range(n_turns):
            for elem in self.elements:
                sigma=elem.track_sigma(sigma)
                s_pos+=elem.length
                
                s_history.append(s_pos)
                beta_history.append(sigma[0,0])
                turn_history.append(turn+1)

                
        return np.array(s_history), np.array(beta_history),np.array(turn_history)
        
# Additional func
def make_sigma_from_twiss(beta0,alpha0,epsilon):
    gamma0=(1+alpha0**2)/beta0
    return epsilon*np.array([[beta0,-alpha0],
                             [-alpha0,gamma0]])


# PARAMETERS
EPSILON=1E-6

#creating FODO lattice
# focus-drift-defocus-drift
bl=Beamline()
bl.add(Quadrupole(f=5.0,name='QF'))
bl.add(Drift(2.0))
bl.add(Quadrupole(f=-5.0,name='QD'))
bl.add(Drift(2.0))

# checking stability

stable,trace=bl.is_stable()
print(f'Stability check')
print(f'Tr(M): {trace:.4f}')
print(f'Stable:{stable}')

if not stable:
    print ("THE LATTICE IS UNSTABLE")
else:
    # periodical Twiss parameters
    beta,alpha,gamma,tune=bl.get_periodic_twiss()
    print(f'PERIODICAL PARAMETERS')
    print(f'beta={beta:.4f} m')
    print(f'alpha={alpha:.4f}')
    print(f'gamma={gamma:.4f} 1/m')
    print(f'Tune Q = {tune:.4f}')

    # check the equation
    print(f'\n Checking beta*gamma-alpha^2={beta*gamma-alpha**2:.6f} (should be 1)')

    # creating a beam with periodic conditions
    sigma0=make_sigma_from_twiss(beta,alpha,EPSILON)

    # tracing for 10 laps
    s,beta_hist,turns=bl.track_sigma_along(sigma0,n_turns=10)

    # visualization
    plt.figure(figsize=(12, 5))
    
    # graph 1: beta-function along the laps
    plt.subplot(1, 2, 1)
    plt.plot(s, beta_hist, 'b-', linewidth=1.5, label='β(s)')
    plt.xlabel('Travel s [м]')
    plt.ylabel('β [м]')
    plt.title('Beta-function on 10 laps')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # graph 2: beta at the start of every lap (const)
    plt.subplot(1, 2, 2)
    cell_length = sum(e.length for e in bl.elements)
    beta_at_start = beta_hist[::len(bl.elements)+1]  # get start of a lap
    plt.plot(range(len(beta_at_start)), beta_at_start, 'ro-', label='beta at lap start')
    plt.axhline(beta, color='green', linestyle='--', label=f'periodic beta={beta:.2f}')
    plt.xlabel('lap N')
    plt.ylabel('β [м]')
    plt.title('Stability')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('day4_stability.png', dpi=150)
    plt.show()


bl_unstable=Beamline()
bl_unstable.add(Quadrupole(f=1.0,name="Qf")) # too strong
bl_unstable.add(Drift(2.0))
bl_unstable.add(Quadrupole(f=-1.0,name='QD')) # too strong
bl_unstable.add(Drift(2.0))

stable,trace=bl_unstable.is_stable()
print(f'\n unstable lattice')
print(f'Tr(M) = {trace:.4f}, Stable: {stable}')

if not stable:
    print('BEAM WILL FALL APART AFTER SEVERAL LAPS')



# ADDITIONAL STUFF FOR FUN
f_values=np.linspace(0.5,10,100)
trace_values=[]
stable_mask=[]

for f in f_values:
    bl_test=Beamline()
    bl_test.add(Quadrupole(f=f,name="QF"))
    bl_test.add(Drift(2.0))
    bl_test.add(Quadrupole(f=-f,name='QD'))
    bl_test.add(Drift(2.0))

    _,trace=bl_test.is_stable()
    trace_values.append(trace)
    stable_mask.append(abs(trace)<2)

plt.figure(figsize=(8, 4))
plt.plot(f_values, trace_values, 'b-', label='Tr(M)')
plt.axhline(2, color='red', linestyle='--', label='Stability border')
plt.axhline(-2, color='red', linestyle='--')
plt.fill_between(f_values, -2, 2, where=stable_mask, alpha=0.3, color='green', label='Stable area (from theory)')
plt.xlabel('focus length f [м]')
plt.ylabel('matrix trace Tr(M)')
plt.title('FODO lattice stability diagram')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('day4_stability_diagram.png', dpi=150)
plt.show()