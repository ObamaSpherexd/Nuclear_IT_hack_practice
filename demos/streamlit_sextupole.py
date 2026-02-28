# Запуск: streamlit run demos/streamlit_sextupole.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from elements import Drift, Quadrupole, Sextupole
from beamline import Beamline

st.set_page_config(page_title="🔮 Sextupole Simulator", layout="wide")
st.title("🔮 Симулятор Секступолей (Нелинейная Оптика)")

st.markdown("""
**Секступоли вносят нелинейность** в динамику пучка. Они используются для:
- 🎯 Хроматической коррекции
- 🎯 Контроля динамической апертуры
- ⚠️ Но могут вызывать резонансы и нестабильности!
""")

# === Сайдбар ===
st.sidebar.header("⚙️ Параметры секступоля")
k2 = st.sidebar.number_input("K₂ секступоля (1/м²)", value=10.0, step=1.0)
sext_length = st.sidebar.number_input("Длина секступоля (м)", value=0.1, step=0.05)

st.sidebar.header("📊 Параметры пучка")
n_particles = st.sidebar.number_input("Число частиц", value=100, step=50)
max_amplitude = st.sidebar.slider("Макс. амплитуда (мм)", 1.0, 20.0, 10.0) * 0.001

# === Сборка решётки ===
bl = Beamline()
bl.add(Quadrupole(k=0.1, length=0.5, name="QF"))
bl.add(Drift(2.0))
bl.add(Sextupole(k2=k2, length=sext_length, name="S1"))
bl.add(Drift(2.0))
bl.add(Quadrupole(k=-0.1, length=0.5, name="QD"))
bl.add(Drift(2.0))

# === Трассировка множества частиц ===
st.subheader("🔬 Трассировка частиц через секступоль")

# Генерируем частицы с разными начальными отклонениями
x0_values = np.linspace(-max_amplitude, max_amplitude, n_particles)
y0_values = np.zeros(n_particles)

final_x = []
final_y = []
lost_mask = []

for i in range(n_particles):
    state = np.array([x0_values[i], 0, y0_values[i], 0])
    
    # Трассируем на 10 оборотов
    stable = True
    for turn in range(10):
        state = bl.track_particle_nonlinear(state)
        if abs(state[0]) > 0.1 or abs(state[2]) > 0.1:  # 10 см апертура
            stable = False
            break
    
    final_x.append(state[0] if stable else np.nan)
    final_y.append(state[2] if stable else np.nan)
    lost_mask.append(not stable)

# === Графики ===
col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    ax1.scatter(x0_values*1000, np.array(final_x)*1000, 
                c=['red' if lost else 'green' for lost in lost_mask],
                s=50, alpha=0.6)
    ax1.set_xlabel('Начальное x [мм]')
    ax1.set_ylabel('Конечное x [мм]')
    ax1.set_title('Карта устойчивости (красный = потеря)')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    stable_fraction = 1 - np.mean(lost_mask)
    ax2.bar(['Устойчивые', 'Потерянные'], 
            [stable_fraction * n_particles, np.sum(lost_mask)],
            color=['green', 'red'])
    ax2.set_ylabel('Число частиц')
    ax2.set_title(f'Доля устойчивых: {stable_fraction:.1%}')
    st.pyplot(fig2)

# === Динамическая апертура ===
st.subheader("📏 Динамическая апертура")

amplitudes, stable = bl.get_dynamic_aperture(n_particles=50, max_amplitude=max_amplitude*2)

fig3, ax3 = plt.subplots()
ax3.plot(amplitudes*1000, stable.astype(int), 'go-', markersize=8)
ax3.set_xlabel('Амплитуда [мм]')
ax3.set_ylabel('Устойчивость (1=да, 0=нет)')
ax3.set_title('Динамическая апертура')
ax3.grid(True, alpha=0.3)
ax3.set_yticks([0, 1])
st.pyplot(fig3)

# === Информация ===
st.markdown("---")
st.info("""
**💡 Физика секступолей:**
- Нелинейное поле: $B_y \\propto x^2 - y^2$
- Пинок угла: $\\Delta x' = -\\frac{1}{2} k_2 L (x^2 - y^2)$
- **Не работает матричный метод!** Нужна трассировка частиц.
- Сильные секступоли уменьшают динамическую апертуру.
""")