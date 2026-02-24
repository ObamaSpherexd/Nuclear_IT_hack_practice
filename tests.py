import numpy as np
import matplotlib.pyplot as plt

# === Импорт классов из Day 3 ===
# (скопируй классы Element, Drift, Quadrupole, Beamline)

class Element:
    def __init__(self, name, length=0.0):
        self.name = name
        self.length = length
    
    def matrix(self):
        raise NotImplementedError
    
    def track_sigma(self, sigma_in):
        M = self.matrix()
        return M @ sigma_in @ M.T

class Drift(Element):
    def __init__(self, L):
        super().__init__(f"Drift_{L:.2f}", L)
        self.L = L
    
    def matrix(self):
        return np.array([[1, self.L], 
                         [0, 1]])

class Quadrupole(Element):
    def __init__(self, f, length=0.0, name="Quad"):
        super().__init__(name, length)
        self.f = f
    
    def matrix(self):
        return np.array([[1, 0], 
                         [-1/self.f, 1]])

class Beamline:
    def __init__(self):
        self.elements = []
    
    def add(self, element):
        self.elements.append(element)
        return self
    
    def one_turn_matrix(self):
        """Вычисляет матрицу полного оборота"""
        M = np.eye(2)
        for elem in self.elements:
            M = elem.matrix() @ M  # Умножение слева (порядок важен!)
        return M
    
    def is_stable(self):
        """Проверяет условие устойчивости"""
        M = self.one_turn_matrix()
        trace = np.trace(M)
        return abs(trace) < 2, trace
    
    def get_periodic_twiss(self):
        """Находит периодические параметры Твисса"""
        M = self.one_turn_matrix()
        trace = np.trace(M)
        
        if abs(trace) >= 2:
            raise ValueError(f"Решётка нестабильна! Tr(M) = {trace:.4f}")
        
        mu = np.arccos(trace / 2)  # Фаза за оборот
        sin_mu = np.sin(mu)
        
        # Периодические Твисс-параметры
        beta = M[0, 1] / sin_mu
        alpha = (M[0, 0] - M[1, 1]) / (2 * sin_mu)
        gamma = -M[1, 0] / sin_mu
        
        # Tune (бетатронное число)
        tune = mu / (2 * np.pi)
        
        return beta, alpha, gamma, tune
    
    def track_sigma_along(self, sigma0, n_turns=1):
        """Трассировка Σ-матрицы на N оборотов"""
        sigma = sigma0.copy()
        s_pos = 0
        cell_length = sum(e.length for e in self.elements)
        
        s_history = [0]
        beta_history = [sigma[0, 0]]
        turn_history = [0]
        
        for turn in range(n_turns):
            for elem in self.elements:
                sigma = elem.track_sigma(sigma)
                s_pos += elem.length
                
                s_history.append(s_pos)
                beta_history.append(sigma[0, 0])
                turn_history.append(turn + 1)
        
        return np.array(s_history), np.array(beta_history), np.array(turn_history)

# === Вспомогательная функция ===
def make_sigma_from_twiss(beta0, alpha0, epsilon):
    gamma0 = (1 + alpha0**2) / beta0
    return epsilon * np.array([[beta0, -alpha0], 
                               [-alpha0, gamma0]])
# === Параметры ===
EPSILON = 1e-6  # Эмиттанс

# === Сборка FODO-ячейки ===
# QF (фокус) -> Drift -> QD (дефокус) -> Drift
bl = Beamline()
bl.add(Quadrupole(f=5.0, name="QF"))
bl.add(Drift(2.0))
bl.add(Quadrupole(f=-5.0, name="QD"))
bl.add(Drift(2.0))

# === Проверка устойчивости ===
stable, trace = bl.is_stable()
print(f"=== Проверка устойчивости ===")
print(f"След матрицы: Tr(M) = {trace:.4f}")
print(f"Устойчива: {stable}")

if not stable:
    print("⚠️  Решётка нестабильна! Измените параметры квадруполей.")
else:
    # === Периодические Твисс-параметры ===
    beta, alpha, gamma, tune = bl.get_periodic_twiss()
    print(f"\n=== Периодические параметры ===")
    print(f"β = {beta:.4f} м")
    print(f"α = {alpha:.4f}")
    print(f"γ = {gamma:.4f} 1/м")
    print(f"Tune Q = {tune:.4f}")
    
    # === Проверка тождества ===
    print(f"\nПроверка βγ - α² = {beta*gamma - alpha**2:.6f} (должно быть 1)")
    
    # === Создаём пучок с периодическими условиями ===
    sigma0 = make_sigma_from_twiss(beta, alpha, EPSILON)
    
    # === Трассировка на 10 оборотов ===
    s, beta_hist, turns = bl.track_sigma_along(sigma0, n_turns=10)
    
    # === Визуализация ===
    plt.figure(figsize=(12, 5))
    
    # График 1: β-функция вдоль оборотов
    plt.subplot(1, 2, 1)
    plt.plot(s, beta_hist, 'b-', linewidth=1.5, label='β(s)')
    plt.xlabel('Путь s [м]')
    plt.ylabel('β [м]')
    plt.title('Бета-функция на 10 оборотах')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # График 2: β в начале каждого оборота (должна быть постоянной)
    plt.subplot(1, 2, 2)
    cell_length = sum(e.length for e in bl.elements)
    beta_at_start = beta_hist[::len(bl.elements)+1]  # Берём начало каждого оборота
    plt.plot(range(len(beta_at_start)), beta_at_start, 'ro-', label='β на старте оборота')
    plt.axhline(beta, color='green', linestyle='--', label=f'Периодическое β={beta:.2f}')
    plt.xlabel('Номер оборота')
    plt.ylabel('β [м]')
    plt.title('Стабильность периодического решения')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('day4_stability.png', dpi=150)
    plt.show()
    