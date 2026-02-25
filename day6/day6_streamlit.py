# ⭐ ЛУЧШИЙ ВАРИАНТ ДЛЯ ХАКАТОНА! ⭐
# Запуск: streamlit run day6_streamlit.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Beam Optics Simulator", layout="wide")

st.title("🚀 Симулятор Пучка Ускорителя")
st.markdown("""
Интерактивная демонстрация оптики пучка в циклическом ускорителе.
Изменяйте параметры магнитов и наблюдайте за β-функцией в реальном времени!
""")

# === Сайдбар с параметрами ===
st.sidebar.header("⚙️ Параметры решётки")

q1_f = st.sidebar.slider("Q1 фокусное расстояние (м)", 1.0, 20.0, 5.0, 0.5)
q2_f = st.sidebar.slider("Q2 фокусное расстояние (м)", -20.0, -1.0, -5.0, 0.5)
drift_l = st.sidebar.slider("Длина дрейфа (м)", 0.5, 5.0, 2.0, 0.5)
epsilon = st.sidebar.number_input("Эмиттанс (м·рад)", value=1e-6, format="%.1e")
beta0 = st.sidebar.number_input("Начальная β (м)", value=10.0, step=1.0)

# === Классы (те же самые) ===
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
        return np.array([[1, self.L], [0, 1]])

class Quadrupole(Element):
    def __init__(self, f, length=0.0, name="Quad"):
        super().__init__(name, length)
        self.f = f
    
    def matrix(self):
        return np.array([[1, 0], [-1/self.f, 1]])

class Beamline:
    def __init__(self):
        self.elements = []
    
    def add(self, element):
        self.elements.append(element)
        return self
    
    def get_beta_along(self, sigma0):
        sigma = sigma0.copy()
        s_positions = [0]
        beta_history = [sigma[0, 0] / epsilon]
        
        for elem in self.elements:
            sigma = elem.track_sigma(sigma)
            s_positions.append(s_positions[-1] + elem.length)
            beta_history.append(sigma[0, 0] / epsilon)
        
        return np.array(s_positions), np.array(beta_history)
    
    def one_turn_matrix(self):
        M = np.eye(2)
        for elem in self.elements:
            M = elem.matrix() @ M
        return M
    
    def is_stable(self):
        M = self.one_turn_matrix()
        trace = np.trace(M)
        return abs(trace) < 2, trace

def make_sigma_from_twiss(beta0, alpha0):
    gamma0 = (1 + alpha0**2) / beta0
    return epsilon * np.array([[beta0, -alpha0], [-alpha0, gamma0]])

# === Основная логика ===
sigma0 = make_sigma_from_twiss(beta0, 0.0)

bl = Beamline()
bl.add(Quadrupole(f=q1_f, name="Q1"))
bl.add(Drift(drift_l))
bl.add(Quadrupole(f=q2_f, name="Q2"))
bl.add(Drift(drift_l))

s, beta = bl.get_beta_along(sigma0)

# === Проверка устойчивости ===
stable, trace = bl.is_stable()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("След матрицы", f"{trace:.4f}")
with col2:
    st.metric("Устойчивость", "✅ Да" if stable else "❌ Нет")
with col3:
    st.metric("Макс. β", f"{max(beta):.2f} м")

if not stable:
    st.error("⚠️ Решётка нестабильна! Пучок разлетится.")

# === Графики ===
col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(s, beta, 'b-', linewidth=2)
    ax1.fill_between(s, 0, beta, alpha=0.3, color='blue')
    ax1.set_xlabel('s [м]')
    ax1.set_ylabel('β [м]')
    ax1.set_title('β-функция вдоль решётки')
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sigma_x = np.sqrt(epsilon * beta) * 1000  # в мм
    ax2.plot(s, sigma_x, 'g-', linewidth=2)
    ax2.fill_between(s, 0, sigma_x, alpha=0.3, color='green')
    ax2.set_xlabel('s [м]')
    ax2.set_ylabel('σₓ [мм]')
    ax2.set_title('Размер пучка')
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

# === Информация ===
st.markdown("---")
st.markdown("### 📊 Информация о пучке")
st.write(f"""
- **Эмиттанс:** {epsilon:.1e} м·рад
- **Начальная β:** {beta0} м
- **Минимальный размер пучка:** {min(sigma_x):.3f} мм
- **Максимальный размер пучка:** {max(sigma_x):.3f} мм
""")

# === Кнопка экспорта ===
if st.button('💾 Скачать данные (CSV)'):
    import pandas as pd
    df = pd.DataFrame({'s [м]': s, 'β [м]': beta, 'σₓ [мм]': sigma_x})
    csv = df.to_csv(index=False)
    st.download_button('Скачать CSV', csv, 'beam_data.csv', 'text/csv')