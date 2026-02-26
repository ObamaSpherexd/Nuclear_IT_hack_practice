"""
Beam Accelerator Simulator Package
Package for modelling beam optics in an accelerator
"""

__version__ = "1.0.0"
__author__ = "Your Team Name"

# Импортируем основные классы для удобного доступа
from elements import Element, Drift, Quadrupole
from beamline import Beamline
from twiss import make_sigma_from_twiss, get_twiss_from_sigma, get_emittance
from matching import match_beamline
from visualization import plot_beta_function, plot_phase_space, plot_beam_envelope

__all__ = [
    'Element', 'Drift', 'Quadrupole',
    'Beamline',
    'make_sigma_from_twiss', 'get_twiss_from_sigma', 'get_emittance',
    'match_beamline',
    'plot_beta_function', 'plot_phase_space', 'plot_beam_envelope'
]