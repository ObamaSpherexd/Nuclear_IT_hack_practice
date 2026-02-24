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
        
    def track(self,state):
        # array (2,N)
        return self.matrix() @ state
    
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
    
    def track_beam(self, beam_state):
        '''
        beam_state: matrix (2, N_particles)
        Returns: history (s, x_mean, x_std) for envelope
        '''
        n_particles=beam_state.shape[1]
        s_pos=0
        
        # for statistics
        s_history=[0]
        x_mean_history=[np.mean(beam_state[0])]
        x_std_history=[np.std(beam_state[0])]
        
        for elem in self.elements:
            beam_state= elem.track(beam_state)
            s_pos+=elem.length
            
            s_history.append(s_pos)
            x_mean_history.append(np.mean(beam_state[0]))
            x_std_history.append(np.std(beam_state[0]))
            
        return np.array(s_history), np.array(x_mean_history), np.array(x_std_history)

# BEAM GENERATION
def generate_gaussian_beam(N, sigma_x, sigma_xp):
    '''
    Generates N particles with gaussian spread
    '''
    x=np.random.normal(0,sigma_x,N)
    xp=np.random.normal(0,sigma_xp,N)
    return np.vstack((x,xp))

# SETTINGS

N_PARTICLES=1000
SIGMA_X=0.001 # initial size
SIGMA_XP=0.0001 # initial spread

# creating a mesh (FODO lattice)
# focus - drift - defocus - drift

bl=Beamline()
bl.add(Quadrupole(f=2.0,name='QF'))
bl.add(Drift(1.0))
bl.add(Quadrupole(f=-2.0,name='QD'))
bl.add(Drift(1.0))

# start

beam=generate_gaussian_beam(N_PARTICLES, SIGMA_X, SIGMA_XP) 
s,x_mean,x_std=bl.track_beam(beam)


# visuals

plt.figure(figsize=(10,5))
plt.fill_between(s, x_mean-x_std, x_mean+x_std, color='blue', alpha=0.3, label='Beam Envelope')
plt.plot(s,x_mean,'b-', label='Centroid')
plt.plot(s,np.zeros_like(s),'k--',alpha=0.5, label='Axis')

plt.xlabel('Path s [m]')
plt.ylabel('Position x [m]')
plt.title('Beam Propagation through FODO Cell')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('day2_beam_envelope.png', dpi=150)
plt.show()

# Проверка: сохранился ли эмиттанс (грубо)?
print(f"Начальный размер: {SIGMA_X*1000:.2f} мм")
print(f"Конечный размер: {x_std[-1]*1000:.2f} мм")
