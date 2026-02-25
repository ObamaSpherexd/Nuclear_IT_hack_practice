import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
    
    def get_sigma_along(self,sigma0):
        '''Returns a Sigma in every lattice point'''
        sigma=sigma0.copy()
        s_positions=[0]
        sigma_history=[sigma0.copy()]

        for elem in self.elements:
            sigma=elem.track_sigma(sigma)
            s_positions.append(s_positions[-1]+elem.length)
            sigma_history.append(sigma.copy())

        return np.array(s_positions), sigma_history
    
    def get_beta_along(self,sigma0,epsilon=1e-6):
        '''Returns beta along the lattice'''
        s_positions,sigma_history=self.get_sigma_along(sigma0)
        beta_history=[sigma[0,0]/epsilon for sigma in sigma_history]
        return s_positions, np.array(beta_history)
    
# Additional func
def make_sigma_from_twiss(beta0,alpha0,epsilon):
    gamma0=(1+alpha0**2)/beta0
    return epsilon*np.array([[beta0,-alpha0],
                             [-alpha0,gamma0]])

# CREATING ANIMATION
def create_beam_animation(beamline,sigma0,epsilon=1e-6,n_frames=100):
    '''Creates a beam transport animation (duh)'''
    s_positions,beta_history=beamline.get_beta_along(sigma0,epsilon)

    # Interpolation for smooth animation

    s_smooth=np.linspace(0,s_positions[-1],n_frames)
    beta_smooth=np.interp(s_smooth,s_positions,beta_history)
    sigma_x_smooth=np.sqrt(epsilon*beta_smooth)

    # Setting the fig
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10,6))
    fig.suptitle('Beam propogation animation', fontsize=14)

    # Graph 1: beta-functuion
    ax1.set_xlim(0,s_positions[-1])
    ax1.set_ylim(0,max(beta_history)*1.2)
    ax1.set_xlabel('s, [m]')
    ax1.set_ylabel('beta [m]')
    ax1.grid(True,alpha=0.3)

    beta_line,=ax1.plot([],[],'b-',linewidth=2, label='beta(s)')
    current_pos=ax1.axvline(0,color='red',linestyle='--',label='Beam')
    ax1.legend(loc='upper right')

    # Graph 2: beam cross-function
    ax2.set_xlim(-5*max(sigma_x_smooth)*1000,5*max(sigma_x_smooth)*1000)
    ax2.set_ylim(-0.5,0.5)
    ax2.set_xlabel('x, mm')
    ax2.set_ylabel('Intensity')
    ax2.grid(True,alpha=0.3)


    # Gaussing beam grid
    x_profile=np.linspace(-5*max(sigma_x_smooth)*1000,5*max(sigma_x_smooth)*1000,200)
    profile_line,=ax2.plot([],[],'g-',linewidth=2)
    text_info=ax2.text(0.02,0.95,'',transform=ax2.transAxes,
                       verticalalignment='top',bbox=dict(boxstyle='round',facecolor='wheat',alpha=0.5))
    
    def init():
        beta_line.set_data([],[])
        profile_line.set_data([],[])
        text_info.set_text('')
        return beta_line,profile_line,text_info
    
    def animate(frame):
        # Initializing beta function
        beta_line.set_data(s_smooth[:frame+1],beta_smooth[:frame+1])

        # updating beam position
        current_pos.set_xdata([s_smooth[frame],s_smooth[frame]])

        # updating beam profile
        sigma_x_mm=sigma_x_smooth[frame]*1000 # convert to mm
        gaussian=np.exp(-x_profile**2/(2*sigma_x_mm**2))
        profile_line.set_data(x_profile,gaussian)

        # Updating text
        text_info.set_text(f's= {s_smooth[frame]:.2f} m \n sigma_x = {sigma_x_mm:.2f} mm')

        return beta_line,profile_line,current_pos,text_info
    
    anim=FuncAnimation(fig,animate,init_func=init,frames=n_frames,
                       interval=50,blit=True)
    return anim,fig

# STARTUP
if __name__=='__main__':
    # parameters
    EPSILON=1E-6
    BETA0=10.0
    ALPHA0=0.0
    sigma0=make_sigma_from_twiss(BETA0,ALPHA0,EPSILON)

    # lattice
    bl=Beamline()
    bl.add(Quadrupole(f=5.0, name='Q1'))
    bl.add(Drift(2.0))
    bl.add(Quadrupole(f=5.0,name='Q2'))
    bl.add(Drift(2.0))
    bl.add(Quadrupole(f=5.0,name='Q3'))

    # animation creation
    anim,fig=create_beam_animation(bl,sigma0,EPSILON,n_frames=100)

    # saving to GIF
    anim.save('beam_animation.gif',writer='pillow',fps=20)

    plt.tight_layout()
    plt.savefig('day6_animation_frame.png',dpi=150)
    plt.show()
    print(' ANIMATION CREATED')
