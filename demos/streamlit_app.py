# ⭐ INTERACTIVE BEAM SIMULATOR FOR HACKATHON ⭐
# Запуск: streamlit run demos/streamlit_app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Добавляем путь к src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from elements import Drift, Quadrupole
from beamline import Beamline
from twiss import make_sigma_from_twiss, get_twiss_from_sigma, get_emittance
from matching import match_beamline
from visualization import plot_beta_function, plot_phase_space, plot_beam_envelope

# === Настройка страницы ===
st.set_page_config(
    page_title="🚀 Beam Optics Simulator",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Заголовок ===
st.title("⚛️ Симулятор Оптики Пучка Ускорителя")
st.markdown("""
**Интерактивная демонстрация моделирования пучка заряженных частиц в циклическом ускорителе.**

Изменяйте параметры магнитов и наблюдайте за β-функцией, размером пучка и фазовым портретом в реальном времени!
""")

# === Сайдбар с параметрами ===
st.sidebar.header("⚙️ Параметры решётки")

# Параметры квадруполей
q1_f = st.sidebar.slider("Q1 фокусное расстояние (м)", 1.0, 20.0, 5.0, 0.5, 
                         help="Положительное значение = фокусировка в X")
q2_f = st.sidebar.slider("Q2 фокусное расстояние (м)", -20.0, -1.0, -5.0, 0.5,
                         help="Отрицательное значение = дефокусировка в X")
drift_l = st.sidebar.slider("Длина дрейфа (м)", 0.5, 5.0, 2.0, 0.5)

# Параметры пучка
st.sidebar.header("📊 Параметры пучка")
epsilon = st.sidebar.number_input("Эмиттанс ε (м·рад)", value=1e-6, format="%.1e", 
                                   help="Нормализованный эмиттанс пучка")
beta0 = st.sidebar.number_input("Начальная β (м)", value=10.0, step=1.0)
alpha0 = st.sidebar.number_input("Начальная α", value=0.0, step=0.5)

# === Создание решётки ===
sigma0 = make_sigma_from_twiss(beta0, alpha0, epsilon)

bl = Beamline()
bl.add(Quadrupole(f=q1_f, name="QF"))
bl.add(Drift(drift_l))
bl.add(Quadrupole(f=q2_f, name="QD"))
bl.add(Drift(drift_l))

# === Проверка устойчивости ===
stable, trace = bl.is_stable()
tune = np.arccos(trace / 2) / (2 * np.pi) if abs(trace) < 2 else 0

# === Метрики ===
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("След матрицы Tr(M)", f"{trace:.4f}")
with col2:
    st.metric("Устойчивость", "✅ Да" if stable else "❌ Нет")
with col3:
    st.metric("Tune Q", f"{tune:.3f}" if stable else "N/A")
with col4:
    s, beta = bl.get_beta_along(sigma0, epsilon)
    st.metric("Макс. размер пучка", f"{max(np.sqrt(epsilon * beta)) * 1000:.2f} мм")

# === Предупреждения ===
if not stable:
    st.error("⚠️ **Решётка нестабильна!** Пучок разлетится через несколько оборотов. Измените параметры квадруполей.")
elif abs(trace) > 1.8:
    st.warning("⚠️ **Близко к границе устойчивости!** Рекомендуется увеличить запас.")

if abs(q1_f) < 2.0 or abs(q2_f) < 2.0:
    st.warning("⚠️ **Слишком сильные квадруполи!** Могут быть проблемы с реализацией в реальном ускорителе.")

# === Вкладки с графиками ===
tab1, tab2, tab3, tab4 = st.tabs(["📈 β-функция", "🔬 Размер пучка", "🌀 Фазовый портрет", "🎯 Matching"])

with tab1:
    st.subheader("Бета-функция вдоль решётки")
    fig1 = plot_beta_function(s, beta, title="β(s)", show=False)
    st.pyplot(fig1)
    
    st.markdown("""
    **Что показывает график:**
    - 📍 Где пучок широкий (большая β)
    - 📍 Где пучок узкий (малая β)
    - 📍 Как квадруполи влияют на фокусировку
    """)

with tab2:
    st.subheader("Размер пучка (огибающая)")
    fig2 = plot_beam_envelope(s, beta, epsilon, title="σₓ(s)", show=False)
    st.pyplot(fig2)
    
    sigma_x = np.sqrt(epsilon * beta) * 1000
    st.markdown(f"""
    **Статистика пучка:**
    - Минимальный размер: **{min(sigma_x):.3f} мм**
    - Максимальный размер: **{max(sigma_x):.3f} мм**
    - Средний размер: **{np.mean(sigma_x):.3f} мм**
    """)

with tab3:
    st.subheader("Фазовый портрет (эллипс Твисса)")
    
    # Показываем эллипс в начале и конце
    sigma_end = bl.track_sigma_to_end(sigma0)
    beta_end, alpha_end, _ = get_twiss_from_sigma(sigma_end, epsilon)
    
    col1, col2 = st.columns(2)
    with col1:
        fig3a = plot_phase_space(beta0, alpha0, epsilon, 
                                  title="На входе", show=False)
        st.pyplot(fig3a)
    with col2:
        fig3b = plot_phase_space(beta_end, alpha_end, epsilon,
                                  title="На выходе", show=False)
        st.pyplot(fig3b)
    
    st.markdown("""
    **Что показывает фазовый портрет:**
    - 📐 Форма эллипса = параметры Твисса
    - 📐 Площадь эллипса = эмиттанс (сохраняется!)
    - 📐 Наклон эллипса = α-параметр
    """)

with tab4:
    st.subheader("Автоматическое согласование (Matching)")
    st.markdown("Подберём параметры квадруполей для получения целевых Твисс-параметров.")
    
    col1, col2 = st.columns(2)
    with col1:
        beta_target = st.number_input("Целевая β (м)", value=5.0, step=0.5)
        alpha_target = st.number_input("Целевая α", value=0.0, step=0.5)
    
    with col2:
        if st.button("🚀 Запустить Matching", type="primary"):
            with st.spinner("Оптимизация..."):
                result = match_beamline(bl, sigma0, beta_target, alpha_target, epsilon=epsilon)
                
                if result['success']:
                    st.success(f"✅ Matching успешен! {result['message']}")
                    st.info(f"Итераций: {result['iterations']}, Потери: {result['final_loss']:.2e}")
                    
                    # Обновляем графики с новыми параметрами
                    s_new, beta_new = bl.get_beta_along(sigma0, epsilon)
                    fig_match = plot_beta_function(s_new, beta_new, 
                                                    title=f"После Matching (β={beta_target}м)",
                                                    show=False)
                    st.pyplot(fig_match)
                    
                    # Показываем новые параметры квадруполей
                    st.markdown("### Новые параметры квадруполей:")
                    strengths = bl.get_quadrupole_strengths()
                    st.write(f"- **Q1:** f = {strengths[0]:.3f} м")
                    st.write(f"- **Q2:** f = {strengths[1]:.3f} м")
                else:
                    st.error(f"❌ Matching не удался: {result['message']}")

# === Информация о проекте ===
st.markdown("---")
st.markdown("""
### 📚 О проекте

**Технологии:**
- Python 3.8+
- NumPy (матричные вычисления)
- Matplotlib (визуализация)
- SciPy (оптимизация)
- Streamlit (веб-интерфейс)

**Физическая модель:**
- Метод матриц переноса (Transfer Matrix Method)
- Параметры Твисса (Courant-Snyder parameters)
- Линейная оптика пучка (paraxial approximation)

**Команда:** Ваша Команда | Хакатон 2024
""")

# === Кнопка экспорта данных ===
st.sidebar.markdown("---")
if st.sidebar.button("💾 Экспортировать данные (CSV)"):
    import pandas as pd
    sigma_x = np.sqrt(epsilon * beta) * 1000
    df = pd.DataFrame({
        's [м]': s,
        'β [м]': beta,
        'σₓ [мм]': sigma_x
    })
    csv = df.to_csv(index=False)
    st.sidebar.download_button('Скачать CSV', csv, 'beam_data.csv', 'text/csv')
    st.sidebar.success("✅ Данные готовы к скачиванию!")