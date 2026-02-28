# Запуск: streamlit run demos/streamlit_app_4d.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from elements import Drift, Quadrupole, Dipole
from beamline import Beamline

st.set_page_config(page_title="🚀 4D Beam Simulator", layout="wide")
st.title("🔬 4D Симулятор Пучка (X + Y)")

# === Сайдбар ===
st.sidebar.header("⚙️ Параметры")
k_qf = st.sidebar.number_input("K квадруполя QF (1/м²)", value=0.1, step=0.05)
k_qd = st.sidebar.number_input("K квадруполя QD (1/м²)", value=-0.1, step=0.05)
quad_length = st.sidebar.number_input("Длина квадруполя (м)", value=0.5, step=0.1)
drift_length = st.sidebar.number_input("Длина дрейфа (м)", value=2.0, step=0.5)

# === Начальные условия ===
st.sidebar.header("📊 Начальные условия")
x0 = st.sidebar.number_input("x₀ (мм)", value=1.0, step=0.5) * 0.001
xp0 = st.sidebar.number_input("x'₀ (мрад)", value=0.0, step=0.1) * 0.001
y0 = st.sidebar.number_input("y₀ (мм)", value=1.0, step=0.5) * 0.001
yp0 = st.sidebar.number_input("y'₀ (мрад)", value=0.0, step=0.1) * 0.001

state_4d = np.array([x0, xp0, y0, yp0])

# === Сборка решётки ===
bl = Beamline()
bl.add(Quadrupole(k=k_qf, length=quad_length, name="QF"))
bl.add(Drift(drift_length))
bl.add(Quadrupole(k=k_qd, length=quad_length, name="QD"))
bl.add(Drift(drift_length))

# === Проверка устойчивости ===
stable, trace_x, trace_y = bl.is_stable_4d()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Tr(X)", f"{trace_x:.4f}")
with col2:
    st.metric("Tr(Y)", f"{trace_y:.4f}")
with col3:
    st.metric("Устойчивость", "✅ Да" if stable else "❌ Нет")

if not stable:
    st.error("⚠️ Решётка нестабильна в одной из плоскостей!")

# === Трассировка ===
s_positions = [0]
x_history = [state_4d[0]]
y_history = [state_4d[2]]

state = state_4d.copy()
for elem in bl.elements:
    state = elem.track_4d(state)
    s_positions.append(s_positions[-1] + elem.length)
    x_history.append(state[0])
    y_history.append(state[2])

# === Графики ===
col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    ax1.plot(s_positions, np.array(x_history)*1000, 'b-', label='X [мм]')
    ax1.plot(s_positions, np.array(y_history)*1000, 'r-', label='Y [мм]')
    ax1.set_xlabel('s [м]')
    ax1.set_ylabel('Позиция [мм]')
    ax1.set_title('Трассировка частицы (X и Y)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    ax2.plot(np.array(x_history)*1000, np.array(y_history)*1000, 'go-')
    ax2.set_xlabel('X [мм]')
    ax2.set_ylabel('Y [мм]')
    ax2.set_title('Проекция траектории (X-Y)')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    st.pyplot(fig2)

st.markdown("---")
st.info("💡 **Квадруполь фокусирует в одной плоскости и дефокусирует в другой!**")