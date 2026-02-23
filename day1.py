import numpy as np
import matplotlib.pyplot as plt

def M_drift(L):
    # 'drift mat'
    return np.array([[1,L],
                     [0,1]])
def M_thin_quad(f):
    # thin lens with focal len f
    return np.array([[1,0],
                    [-1/f,1]])

def track_particle(elements,x0,xp0):
    '''elements: list of tuples(matri, length)
    x0,xp0: init conditions
    return: arrays s,x,xp for graph'''
    state=np.array([x0,xp0])
    s_list,x_list,xp_list=[0],[x0],[xp0]
    s=0
    
    for M,length in elements:
        state=M @ state # matrix x vector
        s+=length
        s_list.append(s)
        x_list.append(state[0])
        xp_list.append(state[1])
    return np.array(s_list), np.array(x_list), np.array(xp_list)

# tests
print('drift 1m, x0=0.01 , xp0=0')
M=M_drift(1.0)
x_final=M@np.array([0.001,0])
print(f'result: x={x_final[0]:.4f} m, xp={x_final[1]:.4f} rad')

print("drift 1 м, x0=0, xp0=0.01 rads")
x_final = M_drift(1.0) @ np.array([0, 0.01])
print(f"result: x={x_final[0]:.4f} м, xp={x_final[1]:.4f} rad")

print(" 2: lens f=2 м, x0=0.02 м, xp0=0")
M = M_thin_quad(2.0)
x_final = M @ np.array([0.02, 0])
print(f"result: x={x_final[0]:.4f} м, xp={x_final[1]:.4f} rad")


#visualizing a sequence drift-lens-drift
elements=[
    (M_drift(0.5),0.5),
    (M_thin_quad(1.0),0),
    (M_drift(0.5),0.5)]
s,x,xp=track_particle(elements, x0=0.01, xp0=0)

plt.figure(figsize=(8, 4))
plt.plot(s, x*1000, 'o-', label='x [мм]')  # Переводим в мм для наглядности
plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Путь s [м]')
plt.ylabel('Отклонение x [мм]')
plt.title('Трассировка частицы: Drift(0.5m) → Quad(f=1m) → Drift(0.5m)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('day1_tracking.png', dpi=150)
plt.show()