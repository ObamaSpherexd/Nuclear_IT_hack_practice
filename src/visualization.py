"""
Visualization module
Contains functions for creating beta-functiong graphs, phasal portraits and a beam envelope
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

def plot_beta_function(
    s: np.ndarray, 
    beta: np.ndarray, 
    title: str = "Beta Function",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Creating beta-function along the lattice
    
    Args:
        s: position along the lattice [м]
        beta: beta function value [м]
        title: title duh
        save_path: saving path (None if not needed)
        show: whether to show the graph or no
    
    Returns:
        Figure as a matplotlib object
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(s, beta, 'b-', linewidth=2, label='beta(s)')
    ax.fill_between(s, 0, beta, alpha=0.3, color='blue')
    ax.set_xlabel('Travel s [м]', fontsize=12)
    ax.set_ylabel('beta [м]', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig

def plot_phase_space(
    beta: float, 
    alpha: float, 
    epsilon: float,
    title: str = "Phase Space Ellipse",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Creating a Twiss ellipse in a phase dimension
    
    Args:
        beta: beta-parameter [м]
        alpha: alpha-parameter
        epsilon: emittance [m*rad]
        title: title duh
        save_path: saving path
        show: show graph
    
    Returns:
        Figure as a matplotlib object
    """
    gamma = (1 + alpha**2) / beta
    
    # ellipse parameters
    theta = np.linspace(0, 2 * np.pi, 200)
    x = np.sqrt(epsilon * beta) * np.cos(theta)
    xp = -np.sqrt(epsilon / beta) * (alpha * np.cos(theta) + np.sin(theta))
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x * 1000, xp * 1000, 'b-', linewidth=2, label='Twiss Ellipse')
    ax.fill(x * 1000, xp * 1000, alpha=0.3, color='blue')
    ax.set_xlabel('x [мм]', fontsize=12)
    ax.set_ylabel("x' [mrad]", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend()
    
    # info about the parameters
    info_text = f'β = {beta:.2f} м\nα = {alpha:.2f}\nε = {epsilon:.2e} м·рад'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig

def plot_beam_envelope(
    s: np.ndarray, 
    beta: np.ndarray, 
    epsilon: float,
    title: str = "Beam Envelope",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Creating beam envelope (±σₓ).
    
    Args:
        s: position along the lattice [м]
        beta: beta-function [м]
        epsilon: emittance [m*rad]
        title: title duh
        save_path: saving path
        show: show the graph
    
    Returns:
        Figure as a matplotlib object
    """
    sigma_x = np.sqrt(epsilon * beta) * 1000  # convert мм
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(s, sigma_x, 'g-', linewidth=2, label='+σₓ')
    ax.plot(s, -sigma_x, 'g-', linewidth=2, label='-σₓ')
    ax.fill_between(s, -sigma_x, sigma_x, alpha=0.3, color='green', label='Beam Envelope')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('travel s [м]', fontsize=12)
    ax.set_ylabel('σₓ [мм]', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig

def plot_matching_comparison(
    s: np.ndarray, 
    beta_before: np.ndarray, 
    beta_after: np.ndarray,
    beta_target: float,
    title: str = "Matching Comparison",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Comparing beta before and after matching
    
    Args:
        s: position along lattice [м]
        beta_before: beta before matching [м]
        beta_after: beta after matching [м]
        beta_target: target beta value
        title: title duh
        save_path: saving path
        show: show the graph
    
    Returns:
        Figure as a matplotlib object
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(s, beta_before, 'r--', linewidth=2, alpha=0.7, label='before matching')
    ax.plot(s, beta_after, 'b-', linewidth=2, label='after matching')
    ax.axhline(beta_target, color='green', linestyle=':', linewidth=2, 
               label=f'target β={beta_target:.2f} м')
    ax.set_xlabel('travel s [м]', fontsize=12)
    ax.set_ylabel('beta [м]', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig

def plot_stability_diagram(
    f_values: np.ndarray, 
    trace_values: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Creating stability diagram
    
    Args:
        f_values: focus lengths values [м]
        trace_values: matrix trace values Tr(M)
        save_path: saving path
        show: show the graph
    
    Returns:
        Figure as a matplotlib object
    """
    stable_mask = np.abs(trace_values) < 2
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(f_values, trace_values, 'b-', linewidth=2, label='Tr(M)')
    ax.axhline(2, color='red', linestyle='--', linewidth=2, label='stability border')
    ax.axhline(-2, color='red', linestyle='--', linewidth=2)
    ax.fill_between(f_values, -2, 2, where=stable_mask, 
                    alpha=0.3, color='green', label='stable area')
    ax.set_xlabel('focus length f [м]', fontsize=12)
    ax.set_ylabel('matrix trace Tr(M)', fontsize=12)
    ax.set_title('stability diagram of a FODO lattice', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig