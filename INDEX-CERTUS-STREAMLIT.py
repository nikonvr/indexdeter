#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CERTUS-INDEX Streamlit Edition v5.0 (PGLOBAL Edition)
"""

import os
import sys
import tempfile
import datetime
import io
import time
import base64
from enum import Enum, auto
from typing import Optional, Dict, Any, Tuple, List, Set, Callable
from threading import Event, RLock
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import scipy.optimize
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# NUMBA CONFIGURATION
# =============================================================================

def _configure_numba_env_once():
    if os.environ.get('_CERTUS_NUMBA_CONFIGURED'):
        return
    cache_dir = os.path.join(tempfile.gettempdir(), 'CERTUS_Numba_Cache')
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['NUMBA_CACHE_DIR'] = cache_dir
    n_cores = max(1, (os.cpu_count() or 4) - 1)
    s_cores = str(n_cores)
    os.environ['NUMBA_NUM_THREADS'] = s_cores
    os.environ['_CERTUS_NUMBA_CONFIGURED'] = '1'

_configure_numba_env_once()
from numba import njit, prange

# =============================================================================
# CONSTANTS
# =============================================================================

LOGO_SVG = """
<div style="display: flex; justify-content: center; margin-bottom: 20px;">
<svg width="600" height="124" viewBox="0 0 820 170" xmlns="http://www.w3.org/2000/svg">
  <g transform="translate(120, 90)">
    <path d="M 70,0 C 70,-40 40,-70 0,-70 C -40,-70 -70,-40 -70,0 C -70,40 -40,70 0,70 C 20,70 40,50 40,50"
          fill="none" stroke="#05103b" stroke-width="30" stroke-linecap="round" transform="rotate(-30)"/>
    <path d="M -90,0 C -80,-90 -70,-50 -60,0 S -40,90 -30,0 S -10,-90 0,0 S 20,90 30,0 S 50,-90 60,0 S 80,90 90,0"
          fill="none" stroke="#00f0ff" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>
  </g>
  <g transform="translate(220, 25)">
    <text x="0" y="70" font-family="Arial, sans-serif" font-weight="900" font-size="80" fill="#05103b" letter-spacing="2">CERTUS</text>
    <text x="0" y="110" font-family="Arial, sans-serif" font-weight="300" font-size="18" fill="#000000" letter-spacing="1.2">THIN FILM OPTIMAL STRATEGY FINDER</text>
    <text x="-10" y="135" font-family="Arial, sans-serif" font-size="14" fill="#333333" letter-spacing="0.5">Calculated Error Reduction Through Unbiased Simulation</text>
  </g>
</svg>
</div>
"""

SMALL_EPSILON:  float = 1e-12
HC_EV_NM: float = 1239.84193
PI: float = np.pi
TWO_PI: float = 6.283185307179586

N_MIN_LIMIT: float = 1.0
N_MAX_LIMIT: float = 10.0
K_MAX_LIMIT: float = 8.0

# Substrates Sellmeier coefficients
SELLMEIER_COEFFS_BY_ID: Dict[int, Tuple[float, float, float, float, float, float]] = {
    0: (0.6961663, 0.0684043**2, 0.4079426, 0.1162414**2, 0.8974794, 9.896161**2),
    1: (1.03961212, 0.00600069867, 0.231792344, 0.0200179144, 1.01046945, 103.560653),
    2: (0.90963095, 0.0047563071, 0.37290409, 0.01621977, 0.92110613, 105.77911),
    3: (1.4313493, 0.0726631**2, 0.65054713, 0.1193242**2, 5.3414021, 18.028251**2),
    4: (0.90110328, 0.0045578115, 0.39734436, 0.016601149, 0.94615601, 111.88593),
}

SUBSTRATE_LIST = ["SiO2", "N-BK7", "D263T eco", "Sapphire", "B270i"]
SUBSTRATE_MIN_LAMBDA = {0: 230.0, 1: 400.0, 2: 360.0, 3: 230.0, 4: 400.0}

SUBSTRATES = {
    "SiO2": {"id": 0, "min_lambda": 230.0},
    "N-BK7": {"id": 1, "min_lambda":  400.0},
    "D263T eco": {"id": 2, "min_lambda": 360.0},
    "Sapphire": {"id":  3, "min_lambda": 230.0},
    "B270i":  {"id": 4, "min_lambda": 400.0}
}

# Frosted glass constants
FROSTED_GLASS_CAUCHY_A = 1.5046
FROSTED_GLASS_CAUCHY_B = 4200.0

# =============================================================================
# ENUMERATIONS
# =============================================================================

class DataType(Enum):
    TRANSMISSION = auto()
    REFLECTION = auto()
    BOTH = auto()

class SubstrateMode(Enum):
    STANDARD = auto()
    FROSTED_GLASS = auto()

# =============================================================================
# DATA CLASSES
# =============================================================================

class TLUParameters:
    """Paramètres Tauc-Lorentz-Urbach"""
    def __init__(self, Eg: float, A: float, E0: float, C:  float, Eu: float, eps_inf: float):
        self.Eg = Eg
        self.A = A
        self.E0 = E0
        self.C = C
        self.Eu = Eu
        self.eps_inf = eps_inf
    
    def to_array(self) -> np.ndarray:
        return np.array([self.Eg, self.A, self.E0, self.C, self.Eu, self.eps_inf])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'TLUParameters': 
        return cls(Eg=arr[0], A=arr[1], E0=arr[2], C=arr[3], Eu=arr[4], eps_inf=arr[5])


@dataclass
class Sample:
    """Échantillon pour PGLOBAL"""
    x: np.ndarray
    y: float
    cluster_id: int = -1
    is_origin: bool = False
    generation:  int = 0
    
    def __lt__(self, other:  'Sample') -> bool:
        return self.y < other.y
    
    def __hash__(self) -> int:
        return id(self)


@dataclass
class Cluster:
    """Cluster pour PGLOBAL"""
    id: int
    seed: Sample
    members: Set[int] = field(default_factory=set)
    local_minimum: Optional[Sample] = None
    search_completed: bool = False


@dataclass(frozen=True)
class PGlobalConfig:
    """Configuration PGLOBAL (immuable)"""
    h_init_ratio: float = 0.05
    h_min:  float = 1e-9
    max_fails_before_shrink: int = 2
    shrink_factor: float = 0.5
    accel_factor: float = 2.0
    max_line_search_steps: int = 50
    alpha:  float = 0.01
    n_samples_per_iter: int = 500
    reduction_ratio: float = 0.1
    local_search_budget: int = 40000
    max_feval: int = 1000000
    max_time: float = 3000.0
    convergence_tol: float = 1e-8
    max_active_clusters: int = 50


# =============================================================================
# NUMBA FUNCTIONS - PHYSICS
# =============================================================================

@njit(cache=True, fastmath=True, parallel=True)
def get_n_substrate_array_by_id_kernel(
    wavelengths_nm: np.ndarray,
    B1: float, C1: float,
    B2: float, C2: float,
    B3: float, C3: float,
    min_lambda: float
) -> np.ndarray:
    """Calcule l'indice de réfraction du substrat via Sellmeier."""
    n = len(wavelengths_nm)
    results = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        wl = wavelengths_nm[i]
        if wl < min_lambda:
            results[i] = np.nan
        else:
            l_um = wl / 1000.0
            l_um_sq = l_um * l_um
            n_sq = 1.0 + (B1 * l_um_sq / (l_um_sq - C1)) + \
                         (B2 * l_um_sq / (l_um_sq - C2)) + \
                         (B3 * l_um_sq / (l_um_sq - C3))
            results[i] = np.sqrt(max(n_sq, 1e-6))
    
    return results


def get_n_substrate_array_by_id(substrate_id: int, wavelengths_nm:  np.ndarray) -> np.ndarray:
    """Calcule l'indice de réfraction du substrat par son ID."""
    coeffs = SELLMEIER_COEFFS_BY_ID[substrate_id]
    min_lambda = SUBSTRATE_MIN_LAMBDA.get(substrate_id, 200.0)
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=np.float64)
    return get_n_substrate_array_by_id_kernel(
        wavelengths_nm, coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5], min_lambda
    )


@njit(cache=True, fastmath=True, parallel=True)
def get_n_frosted_glass_array(wavelengths_nm:  np.ndarray) -> np.ndarray:
    """Calculate frosted glass refractive index."""
    n = len(wavelengths_nm)
    results = np.empty(n, dtype=np.float64)
    for i in prange(n):
        results[i] = FROSTED_GLASS_CAUCHY_A + FROSTED_GLASS_CAUCHY_B / (wavelengths_nm[i] * wavelengths_nm[i])
    return results


@njit(cache=True, fastmath=True, parallel=True)
def epsilon2_TLU_array(E_array: np.ndarray, Eg: float, A: float,
                       E0: float, C: float, Eu: float) -> np.ndarray:
    """Calcule ε₂ via le modèle Tauc-Lorentz-Urbach."""
    n = len(E_array)
    result = np.empty(n, dtype=np.float64)
    
    E0_sq = E0 * E0
    C_sq = C * C
    A_E0_C = A * E0 * C
    
    delta = 0.01
    E_edge = Eg + delta
    E_edge_sq = E_edge * E_edge
    
    num_edge = A_E0_C * delta * delta
    den_edge = E_edge * ((E_edge_sq - E0_sq)**2 + C_sq * E_edge_sq)
    eps2_at_edge = num_edge / den_edge if den_edge > SMALL_EPSILON else 0.0
    
    Eu_safe = max(Eu, 1e-6)
    
    for i in prange(n):
        E = E_array[i]
        
        if E > Eg:
            E_sq = E * E
            diff = E - Eg
            num = A_E0_C * diff * diff
            den = E * ((E_sq - E0_sq)**2 + C_sq * E_sq)
            result[i] = num / den if den > SMALL_EPSILON else 0.0
        else: 
            if eps2_at_edge < SMALL_EPSILON: 
                result[i] = 0.0
            else: 
                result[i] = eps2_at_edge * np.exp((E - Eg - delta) / Eu_safe)
    
    return result


@njit(cache=True, fastmath=True, parallel=True)
def epsilon1_TL_analytic(E_array: np.ndarray, Eg: float, A: float,
                         E0: float, C: float, eps_inf: float) -> np.ndarray:
    """Calcule ε₁ via Kramers-Kronig."""
    n = len(E_array)
    eps1_array = np.empty(n, dtype=np.float64)
    
    E0_sq = E0 * E0
    Eg_sq = Eg * Eg
    C_sq = C * C
    
    gamma_sq = E0_sq - C_sq / 2.0
    alpha = np.sqrt(max(4.0 * E0_sq - C_sq, 1e-12))
    denom_log_norm = np.sqrt((E0_sq - Eg_sq)**2 + C_sq * Eg_sq)
    
    A_E0_C = A * E0 * C
    two_A_E0_C_Eg = 2.0 * A_E0_C * Eg
    inv_PI = 1.0 / PI
    
    for i in prange(n):
        E = E_array[i]
        E_sq = E * E
        
        zeta4 = (E_sq - E0_sq)**2 + C_sq * E_sq
        if zeta4 < SMALL_EPSILON: 
            zeta4 = SMALL_EPSILON
        inv_zeta4 = 1.0 / zeta4
        
        al = (Eg_sq - E0_sq) * E_sq + Eg_sq * C_sq - E0_sq * (E0_sq + 3.0 * Eg_sq)
        aa = (E_sq - E0_sq) * (E0_sq + Eg_sq) + Eg_sq * C_sq
        
        term1 = 0.0
        if E > SMALL_EPSILON: 
            val_log1 = np.log(np.abs((Eg - E) / (Eg + E)))
            term1 = -A_E0_C * (E_sq + Eg_sq) * inv_PI * inv_zeta4 / E * val_log1
        
        val_log2 = np.log(np.abs((Eg - E) * (Eg + E)) / denom_log_norm)
        term2 = two_A_E0_C_Eg * inv_PI * inv_zeta4 * val_log2
        
        arg_log3_num = E0_sq + Eg_sq + alpha * Eg
        arg_log3_den = E0_sq + Eg_sq - alpha * Eg
        if arg_log3_den > SMALL_EPSILON and alpha > SMALL_EPSILON:
            term3 = (A * C * al) / (2.0 * PI * zeta4 * alpha * E0) * np.log(arg_log3_num / arg_log3_den)
        else:
            term3 = 0.0
        
        atan_arg1 = (2.0 * Eg + alpha) / C
        atan_arg2 = (2.0 * Eg - alpha) / C
        term4 = -(A * aa) * inv_PI * inv_zeta4 / E0 * (PI - np.arctan(atan_arg1) - np.arctan(atan_arg2))
        
        atan_arg3 = 2.0 * (Eg_sq - gamma_sq) / max(alpha * C, SMALL_EPSILON)
        if alpha > SMALL_EPSILON:
            term5 = (4.0 * A * E0 * Eg * (E_sq - gamma_sq)) / (PI * zeta4 * alpha) * (PI/2.0 - np.arctan(atan_arg3))
        else:
            term5 = 0.0
        
        val = eps_inf + term1 + term2 + term3 + term4 + term5
        eps1_array[i] = max(val, 1.0) if np.isfinite(val) else eps_inf
    
    return eps1_array


@njit(cache=True, fastmath=True, parallel=True)
def epsilon_to_nk(eps1: np.ndarray, eps2: np.ndarray,
                  n_min: float, n_max: float, k_max: float) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Convertit (ε₁, ε₂) en (n, k)."""
    n_pts = len(eps1)
    n_arr = np.empty(n_pts, dtype=np.float64)
    k_arr = np.empty(n_pts, dtype=np.float64)
    is_valid = True
    
    for i in prange(n_pts):
        e1 = eps1[i]
        e2 = eps2[i]
        eps_mag = np.sqrt(e1 * e1 + e2 * e2)
        
        n_val = np.sqrt(max((eps_mag + e1) / 2.0, SMALL_EPSILON))
        k_val = np.sqrt(max((eps_mag - e1) / 2.0, 0.0))
        
        if n_val < n_min or n_val > n_max or k_val > k_max: 
            is_valid = False
        
        n_arr[i] = n_val
        k_arr[i] = k_val
    
    return n_arr, k_arr, is_valid


@njit(cache=True, fastmath=True)
def calculate_transmission_single(wavelength:  float, n_film_real: float,
                                  n_film_imag: float, thickness_nm: float,
                                  n_sub:  float) -> float:
    """Calcule la transmission d'une monocouche."""
    if not np.isfinite(n_sub) or n_sub < 1.0:
        return np.nan
    
    n0 = 1.0
    k = TWO_PI / wavelength
    
    phi_r = k * n_film_real * thickness_nm
    phi_i = k * n_film_imag * thickness_nm
    
    exp_pos = np.exp(-phi_i)
    exp_neg = np.exp(phi_i)
    
    cos_phi_r = np.cos(phi_r)
    sin_phi_r = np.sin(phi_r)
    
    cos_phi_real = cos_phi_r * (exp_pos + exp_neg) / 2.0
    cos_phi_imag = sin_phi_r * (exp_neg - exp_pos) / 2.0
    sin_phi_real = sin_phi_r * (exp_pos + exp_neg) / 2.0
    sin_phi_imag = cos_phi_r * (exp_pos - exp_neg) / 2.0
    
    n_mag_sq = n_film_real * n_film_real + n_film_imag * n_film_imag
    if n_mag_sq < SMALL_EPSILON:
        return np.nan
    
    inv_n_r = n_film_real / n_mag_sq
    inv_n_i = n_film_imag / n_mag_sq
    
    M01_real = -(inv_n_r * sin_phi_imag + inv_n_i * sin_phi_real)
    M01_imag = inv_n_r * sin_phi_real - inv_n_i * sin_phi_imag
    M10_real = -(n_film_real * sin_phi_imag + n_film_imag * sin_phi_real)
    M10_imag = n_film_real * sin_phi_real - n_film_imag * sin_phi_imag
    
    denom_real = n0 * cos_phi_real + n0 * n_sub * M01_real + M10_real + n_sub * cos_phi_real
    denom_imag = n0 * cos_phi_imag + n0 * n_sub * M01_imag + M10_imag + n_sub * cos_phi_imag
    
    denom_mag_sq = denom_real * denom_real + denom_imag * denom_imag
    if denom_mag_sq < SMALL_EPSILON:
        return np.nan
    
    t_real = 2.0 * n0 * denom_real / denom_mag_sq
    t_imag = -2.0 * n0 * denom_imag / denom_mag_sq
    t_mag_sq = t_real * t_real + t_imag * t_imag
    
    T = (n_sub / n0) * t_mag_sq
    return max(0.0, min(1.0, T))


@njit(cache=True, fastmath=True, parallel=True)
def calculate_transmission_array(wavelengths:  np.ndarray, n_array: np.ndarray,
                                 k_array: np.ndarray, thickness:  float,
                                 n_substrate: np.ndarray) -> np.ndarray:
    """Transmission vectorisée."""
    n_pts = len(wavelengths)
    T_array = np.empty(n_pts, dtype=np.float64)
    
    for i in prange(n_pts):
        T_array[i] = calculate_transmission_single(
            wavelengths[i], n_array[i], k_array[i], thickness, n_substrate[i]
        )
    
    return T_array


@njit(cache=True, fastmath=True, parallel=True)
def calculate_T_substrate_array(wavelengths:  np.ndarray,
                                n_substrate:  np.ndarray) -> np.ndarray:
    """Transmission du substrat seul."""
    n_pts = len(wavelengths)
    T_sub = np.empty(n_pts, dtype=np.float64)
    n0 = 1.0
    
    for i in prange(n_pts):
        ns = n_substrate[i]
        if not np.isfinite(ns) or ns < 1.0:
            T_sub[i] = np.nan
        else: 
            T_sub[i] = 4.0 * n0 * ns / ((n0 + ns) ** 2)
    
    return T_sub


@njit(cache=True, fastmath=True)
def calculate_reflection_single(wavelength: float, n_film_real: float,
                                n_film_imag: float, thickness_nm: float,
                                n_sub: float) -> float:
    """Calcule la réflexion d'une monocouche."""
    if not np.isfinite(n_sub) or n_sub < 1.0:
        return np.nan
    
    n0 = 1.0
    k = TWO_PI / wavelength
    
    phi_r = k * n_film_real * thickness_nm
    phi_i = k * n_film_imag * thickness_nm
    
    exp_pos = np.exp(-phi_i)
    exp_neg = np.exp(phi_i)
    
    cos_phi_r = np.cos(phi_r)
    sin_phi_r = np.sin(phi_r)
    
    cos_phi_real = cos_phi_r * (exp_pos + exp_neg) / 2.0
    cos_phi_imag = sin_phi_r * (exp_neg - exp_pos) / 2.0
    sin_phi_real = sin_phi_r * (exp_pos + exp_neg) / 2.0
    sin_phi_imag = cos_phi_r * (exp_pos - exp_neg) / 2.0
    
    n_mag_sq = n_film_real * n_film_real + n_film_imag * n_film_imag
    if n_mag_sq < SMALL_EPSILON:
        return np.nan
    
    inv_n_r = n_film_real / n_mag_sq
    inv_n_i = n_film_imag / n_mag_sq
    
    M00_real = cos_phi_real
    M00_imag = cos_phi_imag
    M01_real = -(inv_n_r * sin_phi_imag + inv_n_i * sin_phi_real)
    M01_imag = inv_n_r * sin_phi_real - inv_n_i * sin_phi_imag
    M10_real = -(n_film_real * sin_phi_imag + n_film_imag * sin_phi_real)
    M10_imag = n_film_real * sin_phi_real - n_film_imag * sin_phi_imag
    M11_real = cos_phi_real
    M11_imag = cos_phi_imag
    
    denom_real = n0 * M00_real + n0 * n_sub * M01_real + M10_real + n_sub * M11_real
    denom_imag = n0 * M00_imag + n0 * n_sub * M01_imag + M10_imag + n_sub * M11_imag
    
    denom_mag_sq = denom_real * denom_real + denom_imag * denom_imag
    if denom_mag_sq < SMALL_EPSILON:
        return np.nan
    
    num_r_real = n0 * M00_real - n0 * n_sub * M01_real + M10_real - n_sub * M11_real
    num_r_imag = n0 * M00_imag - n0 * n_sub * M01_imag + M10_imag - n_sub * M11_imag
    
    r_real = (num_r_real * denom_real + num_r_imag * denom_imag) / denom_mag_sq
    r_imag = (num_r_imag * denom_real - num_r_real * denom_imag) / denom_mag_sq
    
    R = r_real * r_real + r_imag * r_imag
    return max(0.0, min(1.0, R))


@njit(cache=True, fastmath=True, parallel=True)
def calculate_reflection_array(wavelengths: np.ndarray, n_array: np.ndarray,
                               k_array:  np.ndarray, thickness: float,
                               n_substrate: np.ndarray) -> np.ndarray:
    """Réflexion vectorisée."""
    n_pts = len(wavelengths)
    R_array = np.empty(n_pts, dtype=np.float64)
    
    for i in prange(n_pts):
        R_array[i] = calculate_reflection_single(
            wavelengths[i], n_array[i], k_array[i], thickness, n_substrate[i]
        )
    
    return R_array


@njit(cache=True, fastmath=True, parallel=True)
def calculate_R_substrate_array(wavelengths: np.ndarray,
                                n_substrate: np.ndarray) -> np.ndarray:
    """Réflexion du substrat seul."""
    n_pts = len(wavelengths)
    R_sub = np.empty(n_pts, dtype=np.float64)
    n0 = 1.0
    
    for i in prange(n_pts):
        ns = n_substrate[i]
        if not np.isfinite(ns) or ns < 1.0:
            R_sub[i] = np.nan
        else:
            r = (n0 - ns) / (n0 + ns)
            R_sub[i] = r * r
    
    return R_sub


@njit(cache=True, fastmath=True, parallel=True)
def calculate_reflection_infinite_substrate_array(
    wavelengths: np.ndarray,
    n_array: np.ndarray,
    k_array: np.ndarray,
    thickness: float,
    n_substrate: np.ndarray
) -> np.ndarray:
    """Réflexion pour substrat infini (frosted glass)."""
    n_pts = len(wavelengths)
    R_array = np.empty(n_pts, dtype=np.float64)
    
    for i in prange(n_pts):
        R_array[i] = calculate_reflection_single(
            wavelengths[i], n_array[i], k_array[i], thickness, n_substrate[i]
        )
    
    return R_array


@njit(cache=True, fastmath=True, parallel=True)
def calculate_R_frosted_glass_reference(wavelengths:  np.ndarray, n_substrate: np.ndarray) -> np.ndarray:
    """Réflexion de référence du frosted glass."""
    n_pts = len(wavelengths)
    R_ref = np.empty(n_pts, dtype=np.float64)
    n0 = 1.0
    
    for i in prange(n_pts):
        n_sub = n_substrate[i]
        r = (n0 - n_sub) / (n0 + n_sub)
        R_ref[i] = r * r
    
    return R_ref


@njit(cache=True, fastmath=True)
def compute_mse_vectorized(calc_values: np.ndarray, target_values: np.ndarray,
                           weights:  np.ndarray) -> Tuple[float, int]:
    """Calcule le MSE pondéré."""
    n = len(calc_values)
    sum_sq = 0.0
    count = 0
    
    for i in range(n):
        if weights[i] > 0 and np.isfinite(calc_values[i]) and np.isfinite(target_values[i]):
            diff = calc_values[i] - target_values[i]
            sum_sq += diff * diff * weights[i]
            count += 1
    
    if count < 5:
        return np.inf, count
    
    return sum_sq / count, count


# =============================================================================
# NUMBA FUNCTIONS - OPTIMIZATION UTILITIES
# =============================================================================

@njit(cache=True, fastmath=True)
def clip_to_bounds(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Clip paramètres aux bornes"""
    result = np.empty_like(x)
    for i in range(len(x)):
        if x[i] < lb[i]:
            result[i] = lb[i]
        elif x[i] > ub[i]:
            result[i] = ub[i]
        else:
            result[i] = x[i]
    return result


@njit(cache=True, fastmath=True)
def norm_inf(x: np.ndarray, y: np.ndarray) -> float:
    """Norme infinie (distance de Chebyshev)"""
    d = 0.0
    for i in range(len(x)):
        diff = abs(x[i] - y[i])
        if diff > d: 
            d = diff
    return d


@njit(cache=True, fastmath=True)
def generate_normal_direction(dim: int) -> np.ndarray:
    """Génère une direction aléatoire unitaire"""
    d = np.random.standard_normal(dim)
    norm = 0.0
    for i in range(dim):
        norm += d[i] * d[i]
    norm = np.sqrt(norm)
    if norm > 1e-12:
        for i in range(dim):
            d[i] /= norm
    return d


@njit(cache=True, fastmath=True, nogil=True)
def compute_critical_distance(N: int, n:  int, alpha: float) -> float:
    """Calcule la distance critique pour le clustering."""
    if N <= 1: 
        return 1.0
    exponent = 1.0 / (N - 1)
    base = 1.0 - alpha ** exponent
    if base <= 0:
        return 0.0
    return base ** (1.0 / n)


@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def fast_clustering_kernel(x_candidates: np.ndarray, y_candidates:  np.ndarray,
                           x_seeds: np.ndarray, y_seeds:  np.ndarray,
                           dc: float) -> np.ndarray:
    """Clustering rapide via distance de Chebyshev."""
    n_cand = len(x_candidates)
    n_seeds = len(x_seeds)
    cluster_ids = np.full(n_cand, -1, dtype=np.int32)
    
    if n_seeds == 0:
        return cluster_ids
    
    for i in prange(n_cand):
        xc = x_candidates[i]
        for j in range(n_seeds):
            dist = 0.0
            xs = x_seeds[j]
            for k in range(len(xc)):
                d = abs(xc[k] - xs[k])
                if d > dist:
                    dist = d
            if dist <= dc:
                cluster_ids[i] = j
                break
    
    return cluster_ids


# =============================================================================
# COORDINATE DESCENT
# =============================================================================

@njit(cache=True, fastmath=True)
def run_coordinate_descent_RT(start_params: np.ndarray, wavelengths: np.ndarray,
                              target_T: np.ndarray, target_R: np.ndarray,
                              n_substrate: np.ndarray, weights: np.ndarray,
                              use_T:  bool, use_R: bool, use_normalized: bool,
                              weight_T: float, weight_R: float,
                              is_frosted_glass: bool,
                              max_iter: int) -> Tuple[np.ndarray, float]:
    """Descente de coordonnées pour R, T ou R+T."""
    current_params = start_params.copy()
    n_params = len(current_params)
    
    steps = np.array([2.0, 0.02, 0.5, 0.02, 0.02, 0.01, 0.05], dtype=np.float64)
    min_steps = np.array([1e-4, 1e-6, 1e-5, 1e-6, 1e-6, 1e-6, 1e-5], dtype=np.float64)
    
    E_array = HC_EV_NM / wavelengths
    n_wls = len(wavelengths)
    
    T_substrate = calculate_T_substrate_array(wavelengths, n_substrate)
    
    if is_frosted_glass:
        R_substrate = calculate_R_frosted_glass_reference(wavelengths, n_substrate)
    else:
        R_substrate = calculate_R_substrate_array(wavelengths, n_substrate)
    
    def evaluate_params(p):
        d = p[0]
        p_Eg, p_A, p_E0, p_C, p_Eu, p_eps = p[1], p[2], p[3], p[4], p[5], p[6]
        
        if d < 10 or p_Eg < 0.1 or p_A < 0.1 or p_E0 < p_Eg or p_C < 0.01 or p_Eu < 0.001 or p_eps < 0.5:
            return 1e12
        
        e2 = epsilon2_TLU_array(E_array, p_Eg, p_A, p_E0, p_C, p_Eu)
        e1 = epsilon1_TL_analytic(E_array, p_Eg, p_A, p_E0, p_C, p_eps)
        nc, kc, v = epsilon_to_nk(e1, e2, 0.5, 10.0, 10.0)
        
        if not v:
            return 1e12
        
        total_mse = 0.0
        total_weight = 0.0
        
        if use_T and not is_frosted_glass:
            Tc = calculate_transmission_array(wavelengths, nc, kc, d, n_substrate)
            if use_normalized:
                for k_idx in range(n_wls):
                    if T_substrate[k_idx] < 1e-6:
                        Tc[k_idx] = np.nan
                    else:
                        Tc[k_idx] = Tc[k_idx] / T_substrate[k_idx]
            mse_t, n_t = compute_mse_vectorized(Tc, target_T, weights)
            if n_t >= 5:
                total_mse += mse_t * weight_T
                total_weight += weight_T
        
        if use_R: 
            if is_frosted_glass:
                Rc = calculate_reflection_infinite_substrate_array(wavelengths, nc, kc, d, n_substrate)
            else: 
                Rc = calculate_reflection_array(wavelengths, nc, kc, d, n_substrate)
            
            if use_normalized: 
                for k_idx in range(n_wls):
                    if R_substrate[k_idx] < 1e-6:
                        Rc[k_idx] = np.nan
                    else:
                        Rc[k_idx] = Rc[k_idx] / R_substrate[k_idx]
            mse_r, n_r = compute_mse_vectorized(Rc, target_R, weights)
            if n_r >= 5:
                total_mse += mse_r * weight_R
                total_weight += weight_R
        
        if total_weight < 1e-9:
            return 1e12
        
        return total_mse / total_weight
    
    best_mse = evaluate_params(current_params)
    
    for iteration in range(max_iter):
        for i in range(n_params):
            if steps[i] < min_steps[i]:
                continue
            
            original_val = current_params[i]
            step = steps[i]
            
            current_params[i] = original_val + step
            mse_plus = evaluate_params(current_params)
            
            if mse_plus < best_mse: 
                best_mse = mse_plus
                steps[i] *= 1.2
                continue
            
            current_params[i] = original_val - step
            mse_minus = evaluate_params(current_params)
            
            if mse_minus < best_mse:
                best_mse = mse_minus
                steps[i] *= 1.2
            else:
                current_params[i] = original_val
                steps[i] *= 0.5
        
        all_small = True
        for i in range(n_params):
            if steps[i] >= min_steps[i]:
                all_small = False
                break
        if all_small: 
            break
    
    return current_params, best_mse


# =============================================================================
# DATA TYPE DETECTION
# =============================================================================

def detect_data_type(data: np.ndarray, threshold: float = 0.80) -> str:
    if len(data) == 0: return 'T'  # Sécurité pour tableau vide
    """Détecte si les données sont T ou R."""
    data_copy = data.copy()
    
    if np.nanmax(data_copy) > 1.5:
        data_copy = data_copy / 100.0
    
    valid_data = data_copy[np.isfinite(data_copy)]
    if len(valid_data) == 0:
        return 'T'
    
    n_above_threshold = np.sum(valid_data > threshold)
    ratio_above = n_above_threshold / len(valid_data)
    
    if ratio_above > 0.10:
        return 'T'
    
    if np.nanmax(valid_data) > 0.95:
        return 'T'
    
    return 'R'


def analyze_loaded_data(df: pd.DataFrame) -> Tuple[DataType, Dict[str, np.ndarray]]: 
    """Analyse un DataFrame pour détecter le type de données."""
    result = {
        'lambda': df.iloc[: , 0].to_numpy().astype(np.float64),
        'T': None,
        'R': None
    }
    
    n_cols = len(df.columns)
    
    if n_cols == 2:
        col2_data = df.iloc[:, 1].to_numpy().astype(np.float64)
        col2_type = detect_data_type(col2_data)
        
        if col2_type == 'T':
            result['T'] = col2_data
            return DataType.TRANSMISSION, result
        else:
            result['R'] = col2_data
            return DataType.REFLECTION, result
    
    elif n_cols >= 3:
        col2_data = df.iloc[:, 1].to_numpy().astype(np.float64)
        col3_data = df.iloc[:, 2].to_numpy().astype(np.float64)
        
        col2_type = detect_data_type(col2_data)
        col3_type = detect_data_type(col3_data)
        
        if col2_type == col3_type: 
            if col2_type == 'T': 
                result['T'] = col2_data
                return DataType.TRANSMISSION, result
            else: 
                result['R'] = col2_data
                return DataType.REFLECTION, result
        
        if col2_type == 'T':
            result['T'] = col2_data
            result['R'] = col3_data
        else:
            result['R'] = col2_data
            result['T'] = col3_data
        
        return DataType.BOTH, result
    
    result['T'] = df.iloc[:, 1].to_numpy().astype(np.float64) if n_cols > 1 else np.array([])
    return DataType.TRANSMISSION, result


# =============================================================================
# TLU OBJECTIVE FUNCTION
# =============================================================================

class TLUObjective:
    """Fonction objectif pour l'optimisation TLU."""

    def __init__(self, wavelengths: np.ndarray,
                 target_T: Optional[np.ndarray],
                 target_R: Optional[np.ndarray],
                 n_substrate: np.ndarray,
                 data_type: DataType,
                 thickness_bounds: tuple,
                 use_normalized: bool = True,
                 weight_T: float = 1.0,
                 weight_R: float = 1.0,
                 exclude_range: Optional[tuple] = None,
                 is_frosted_glass: bool = False):
        
        self.wavelengths = wavelengths.astype(np.float64)
        self.n_substrate = n_substrate.astype(np.float64)
        self.data_type = data_type
        self.thickness_bounds = thickness_bounds
        self.use_normalized = use_normalized
        self.weight_T = weight_T
        self.weight_R = weight_R
        self.is_frosted_glass = is_frosted_glass
        self.E_array = HC_EV_NM / self.wavelengths

        self.target_T = target_T.astype(np.float64) if target_T is not None else None
        self.target_R = target_R.astype(np.float64) if target_R is not None else None

        self.T_substrate = calculate_T_substrate_array(self.wavelengths, self.n_substrate)
        
        if is_frosted_glass:
            self.R_substrate = calculate_R_frosted_glass_reference(self.wavelengths, self.n_substrate)
        else:
            self.R_substrate = calculate_R_substrate_array(self.wavelengths, self.n_substrate)

        self.weights = np.ones_like(wavelengths)
        if exclude_range: 
            ex_min, ex_max = exclude_range
            mask = (self.wavelengths >= ex_min) & (self.wavelengths <= ex_max)
            self.weights[mask] = 0.0

        self.n_evals = 0
        self.best_value = np.inf
        self.best_params = None

        self.param_bounds = np.array([
            [0.5, 6.0],
            [10.0, 2000.0],
            [1.5, 10.0],
            [0.1, 10.0],
            [0.01, 3.0],
            [1.0, 10.0]
        ])

    def get_bounds(self) -> np.ndarray:
        thickness_bound = np.array([[self.thickness_bounds[0], self.thickness_bounds[1]]])
        return np.vstack([thickness_bound, self.param_bounds])

    def __call__(self, params: np.ndarray) -> float:
        self.n_evals += 1

        try:
            thickness = params[0]
            Eg, A, E0, C, Eu, eps_inf = params[1:7]
            penalty = 0.0

            if thickness < self.thickness_bounds[0]: 
                penalty += 1000 * (self.thickness_bounds[0] - thickness) ** 2
            elif thickness > self.thickness_bounds[1]: 
                penalty += 1000 * (thickness - self.thickness_bounds[1]) ** 2

            if E0 <= Eg:
                penalty += 100 * (Eg - E0 + 0.1) ** 2
            elif E0 - Eg < 0.3:
                penalty += 10 * (0.3 - (E0 - Eg)) ** 2

            if Eg <= 0 or A <= 0 or C <= 0 or Eu <= 0 or eps_inf < 1: 
                return 1e12 + penalty

            eps2 = epsilon2_TLU_array(self.E_array, Eg, A, E0, C, Eu)
            eps1 = epsilon1_TL_analytic(self.E_array, Eg, A, E0, C, eps_inf)
            n_calc, k_calc, is_valid = epsilon_to_nk(
                eps1, eps2, N_MIN_LIMIT, N_MAX_LIMIT, K_MAX_LIMIT
            )

            if not is_valid: 
                return 1e12 + penalty

            total_mse = 0.0
            total_weight = 0.0

            if (self.data_type in (DataType.TRANSMISSION, DataType.BOTH)
                and self.target_T is not None
                and not self.is_frosted_glass):
                
                T_calc = calculate_transmission_array(
                    self.wavelengths, n_calc, k_calc, thickness, self.n_substrate
                )
                
                if self.use_normalized:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        T_sub_safe = np.where(self.T_substrate > SMALL_EPSILON, self.T_substrate, 1.0)
                        T_val = T_calc / T_sub_safe
                        T_val = np.where(self.T_substrate > SMALL_EPSILON, T_val, np.nan)
                else:
                    T_val = T_calc
                
                mse_T, n_valid_T = compute_mse_vectorized(T_val, self.target_T, self.weights)
                if n_valid_T >= 5:
                    total_mse += mse_T * self.weight_T
                    total_weight += self.weight_T

            if (self.data_type in (DataType.REFLECTION, DataType.BOTH)
                and self.target_R is not None):
                
                if self.is_frosted_glass:
                    R_calc = calculate_reflection_infinite_substrate_array(
                        self.wavelengths, n_calc, k_calc, thickness, self.n_substrate
                    )
                else: 
                    R_calc = calculate_reflection_array(
                        self.wavelengths, n_calc, k_calc, thickness, self.n_substrate
                    )
                
                if self.use_normalized:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        R_sub_safe = np.where(self.R_substrate > SMALL_EPSILON, self.R_substrate, 1.0)
                        R_val = R_calc / R_sub_safe
                        R_val = np.where(self.R_substrate > SMALL_EPSILON, R_val, np.nan)
                else:
                    R_val = R_calc
                
                mse_R, n_valid_R = compute_mse_vectorized(R_val, self.target_R, self.weights)
                if n_valid_R >= 5:
                    total_mse += mse_R * self.weight_R
                    total_weight += self.weight_R

            if total_weight < SMALL_EPSILON: 
                return 1e12 + penalty

            mse = total_mse / total_weight
            total = mse + penalty

            if total < self.best_value:
                self.best_value = total
                self.best_params = params.copy()

            return total

        except Exception: 
            return 1e12


# =============================================================================
# L-BFGS-B SEARCHER
# =============================================================================

def get_lbfgsb_params(dim: int) -> dict:
    """Retourne les paramètres L-BFGS-B adaptés à la dimension."""
    return {
        'ftol': 1e-12,
        'gtol': 1e-12,
        'maxcor': min(50, max(20, dim + 5)),
    }


class LBFGSBSearcher: 
    """Recherche locale avec L-BFGS-B"""
    
    def __init__(self, func:  Callable, bounds: np.ndarray,
                 config: Optional[PGlobalConfig] = None):
        self.func = func
        self.bounds = list(zip(bounds[: , 0], bounds[:, 1]))
        self.dim = len(bounds)
        self.config = config
    
    def search(self, x0: np.ndarray, max_feval: int = 1000) -> Tuple[np.ndarray, float, int]:
        """Effectue une recherche locale."""
        try:
            params = get_lbfgsb_params(self.dim)
            res = scipy.optimize.minimize(
                self.func, x0, method='L-BFGS-B', bounds=self.bounds,
                options={
                    **params,
                    'maxfun':  max_feval,
                    'maxiter': max(100, max_feval // (self.dim + 1))
                }
            )
            return res.x, float(res.fun), int(res.nfev)
        except Exception as e:
            return x0.copy(), float(self.func(x0)), 1


# =============================================================================
# N-UNIR LOCAL SEARCH
# =============================================================================

class NUnirSearcher:
    """N-UNIR local search for refinement"""

    def __init__(self, func, bounds:  np.ndarray, config: PGlobalConfig,
                 stop_event: Optional[Event] = None):
        self.func = func
        self.bounds = bounds
        self.lb = bounds[: , 0]
        self.ub = bounds[:, 1]
        self.dim = len(bounds)
        self.config = config
        self.h_init = config.h_init_ratio * np.mean(self.ub - self.lb)
        self.stop_event = stop_event

    def _safe_eval(self, x:  np.ndarray) -> float:
        if self.stop_event and self.stop_event.is_set():
            return np.inf
        try:
            val = float(self.func(x))
            return val if np.isfinite(val) else np.inf
        except Exception:
            return np.inf

    def _line_search(self, x_start: np.ndarray, f_start: float,
                     direction: np.ndarray, h: float) -> tuple: 
        x_curr = x_start.copy()
        f_curr = f_start
        step = h
        n_evals = 0

        for _ in range(self.config.max_line_search_steps):
            if self.stop_event and self.stop_event.is_set():
                break
            x_trial = clip_to_bounds(x_curr + step * direction, self.lb, self.ub)
            f_trial = self._safe_eval(x_trial)
            n_evals += 1
            if f_trial < f_curr:
                x_curr = x_trial
                f_curr = f_trial
                step *= self.config.accel_factor
            else: 
                break
        return x_curr, f_curr, n_evals

    def search(self, x0: np.ndarray, max_feval: int = 1500) -> tuple:
        x_best = clip_to_bounds(x0.copy(), self.lb, self.ub)
        f_best = self._safe_eval(x_best)
        n_evals = 1
        h = self.h_init
        fails = 0

        while n_evals < max_feval and h > self.config.h_min: 
            if self.stop_event and self.stop_event.is_set():
                break

            for _ in range(self.config.max_fails_before_shrink):
                if n_evals >= max_feval: 
                    break
                if self.stop_event and self.stop_event.is_set():
                    break

                d = generate_normal_direction(self.dim)
                x_trial = clip_to_bounds(x_best + h * d, self.lb, self.ub)
                f_trial = self._safe_eval(x_trial)
                n_evals += 1
                improved = False

                if f_trial < f_best: 
                    x_new, f_new, ne = self._line_search(x_trial, f_trial, d, h)
                    n_evals += ne
                    if f_new < f_best:
                        x_best = x_new
                        f_best = f_new
                        improved = True
                        fails = 0
                else:
                    x_trial = clip_to_bounds(x_best - h * d, self.lb, self.ub)
                    f_trial = self._safe_eval(x_trial)
                    n_evals += 1
                    if f_trial < f_best:
                        x_new, f_new, ne = self._line_search(x_trial, f_trial, -d, h)
                        n_evals += ne
                        if f_new < f_best: 
                            x_best = x_new
                            f_best = f_new
                            improved = True
                            fails = 0

                if not improved: 
                    fails += 1
                if fails >= 2:
                    h *= self.config.shrink_factor
                    fails = 0

        return x_best, f_best, n_evals


# =============================================================================
# SINGLE LINKAGE CLUSTERER
# =============================================================================

class SingleLinkageClusterer: 
    """Clustering single-linkage pour PGLOBAL"""
    
    def __init__(self, dim: int, config: PGlobalConfig):
        self.dim = dim
        self.config = config
        self.clusters:  List[Dict] = []
        self.x_seeds_cache = np.zeros((0, dim), dtype=np.float64)
        self.y_seeds_cache = np.zeros(0, dtype=np.float64)
        self._lock = RLock()
    
    def process_batch(self, x_batch: np.ndarray, y_batch:  np.ndarray,
                      n_total_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Traite un lot de points via Numba."""
        if len(x_batch) == 0:
            return np.array([]).reshape(0, self.dim), np.array([])

        with self._lock:
            effective_alpha = self.config.alpha
            if self.dim > 15:
                effective_alpha = min(0.1, self.config.alpha * 2)
            
            dc = compute_critical_distance(n_total_samples, self.dim, effective_alpha)
            dc = max(dc, 0.01 / np.sqrt(self.dim))
            
            cluster_ids = fast_clustering_kernel(
                x_batch, y_batch,
                self.x_seeds_cache, self.y_seeds_cache,
                dc
            )
            
            mask_unclustered = (cluster_ids == -1)
            candidates_x = x_batch[mask_unclustered]
            candidates_y = y_batch[mask_unclustered]
            
            max_candidates = min(50, max(10, 100 - self.dim))
            if len(candidates_y) > max_candidates:
                idx_best = np.argpartition(candidates_y, max_candidates)[:max_candidates]
                candidates_x = candidates_x[idx_best]
                candidates_y = candidates_y[idx_best]
            
            return candidates_x, candidates_y

    def add_cluster(self, x_local: np.ndarray, y_local:  float):
        """Enregistre un nouveau minimum local comme graine de cluster."""
        with self._lock:
            if len(self.x_seeds_cache) > 0:
                diffs = np.abs(self.x_seeds_cache - x_local)
                dists = np.max(diffs, axis=1)
                
                if np.any(dists < 1e-4):
                    idx_closest = np.argmin(dists)
                    if y_local < self.y_seeds_cache[idx_closest]: 
                        self.x_seeds_cache[idx_closest] = x_local.copy()
                        self.y_seeds_cache[idx_closest] = y_local
                        self.clusters[idx_closest]['x'] = x_local.copy()
                        self.clusters[idx_closest]['y'] = y_local
                    return
            
            self.clusters.append({'x': x_local.copy(), 'y': y_local})
            
            if len(self.x_seeds_cache) > 0:
                self.x_seeds_cache = np.vstack([self.x_seeds_cache, x_local])
            else:
                self.x_seeds_cache = np.atleast_2d(x_local.copy())
            
            self.y_seeds_cache = np.append(self.y_seeds_cache, y_local)

    def get_best_minimum(self) -> Optional[Tuple[np.ndarray, float]]:
        """Retourne le meilleur minimum trouvé"""
        with self._lock:
            if len(self.y_seeds_cache) == 0:
                return None
            idx_best = np.argmin(self.y_seeds_cache)
            return self.x_seeds_cache[idx_best].copy(), float(self.y_seeds_cache[idx_best])
    
    @property
    def n_clusters(self) -> int:
        """Nombre de clusters"""
        return len(self.clusters)
    
    def clear(self):
        """Réinitialise le clusterer"""
        with self._lock:
            self.clusters.clear()
            self.x_seeds_cache = np.zeros((0, self.dim), dtype=np.float64)
            self.y_seeds_cache = np.zeros(0, dtype=np.float64)


# =============================================================================
# PGLOBAL OPTIMIZER
# =============================================================================

class PGlobalOptimizer: 
    """Optimiseur global PGLOBAL complet."""
    
    def __init__(self, objective:  Callable, bounds: np.ndarray,
                 config: Optional[PGlobalConfig] = None,
                 stop_event: Optional[Event] = None,
                 log_indices: Optional[List[int]] = None,
                 n_workers: int = 4):
        self.objective = objective
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.dim = len(bounds)
        self.lb = self.bounds[: , 0]
        self.ub = self.bounds[:, 1]
        self.config = config or PGlobalConfig()
        self.clusterer = SingleLinkageClusterer(self.dim, self.config)
        self._all_samples:  List[Sample] = []
        self.n_evals = 0
        self._n_total_samples = 0
        self._stop_event = stop_event
        self.stats = {'samples': 0, 'local_searches': 0, 'clusters': 0}
        self.log_indices = log_indices or []
        self._evals_lock = RLock()
        self._best_ever:  Optional[Sample] = None
        self.n_workers = n_workers
        self._executor:  Optional[ThreadPoolExecutor] = None
        

    def _increment_evals(self, n:  int):
        """Incrémente le compteur d'évaluations de manière thread-safe"""
        with self._evals_lock:
            self.n_evals += n

    def _should_stop(self) -> bool:
        """Vérifie si l'optimisation doit s'arrêter"""
        if self._stop_event and self._stop_event.is_set():
            return True
        if self.n_evals >= self.config.max_feval:
            return True
        return False

    def _generate_samples(self, n_samples: int) -> np.ndarray:
        """Génère des échantillons aléatoires"""
        X_batch = np.empty((n_samples, self.dim), dtype=np.float64)
        for i in range(self.dim):
            if i in self.log_indices:
                log_lb = np.log10(max(self.lb[i], 1e-9))
                log_ub = np.log10(self.ub[i])
                X_batch[:, i] = np.power(10, np.random.uniform(log_lb, log_ub, n_samples))
            else:
                X_batch[:, i] = np.random.uniform(self.lb[i], self.ub[i], n_samples)
        return X_batch

    def _evaluate_batch(self, X_batch: np.ndarray) -> np.ndarray:
        """Évalue un lot de points"""
        n = len(X_batch)
        Y_batch = np.empty(n, dtype=np.float64)
        for i in range(n):
            if self._should_stop():
                Y_batch[i: ] = np.inf
                break
            val = self.objective(X_batch[i])
            Y_batch[i] = val if np.isfinite(val) else 1e30
        return Y_batch

    def optimize(self, max_iter: int = 50,
                 callback: Optional[Callable[[Sample], None]] = None) -> Optional[Sample]: 
        """Exécute l'optimisation PGLOBAL."""
        start_time = time.time()
        best_ever_y = float('inf')
        best_ever_x = None
        stagnation_counter = 0
        last_best_y = float('inf')
        n_samples_iter = self.config.n_samples_per_iter

        self._executor = ThreadPoolExecutor(max_workers=self.n_workers)

        try:
            for iteration in range(max_iter):
                if self._should_stop():
                    break
                if time.time() - start_time > self.config.max_time:
                    break

                # Échantillonnage adaptatif
                adaptive_factor = 1.0 + 0.5 * (1.0 - iteration / max_iter)
                current_n = int(n_samples_iter * adaptive_factor) if iteration < 10 else n_samples_iter

                # Plus d'échantillons à la première itération
                if iteration == 0:
                    current_n = int(current_n * 2)

                # Génération et évaluation
                X_batch = self._generate_samples(current_n)
                Y_batch = self._evaluate_batch(X_batch)

                self._increment_evals(current_n)
                self._n_total_samples += current_n
                self.stats['samples'] = self._n_total_samples

                # Réduction
                effective_ratio = self.config.reduction_ratio
                if self.dim > 15:
                    effective_ratio = min(0.25, effective_ratio * 1.5)

                n_keep = max(int(current_n * effective_ratio), 10)
                if n_keep < len(Y_batch):
                    idx_best = np.argpartition(Y_batch, n_keep)[: n_keep]
                else:
                    idx_best = np.arange(len(Y_batch))

                # Clustering
                cand_x, cand_y = self.clusterer.process_batch(
                    X_batch[idx_best], Y_batch[idx_best], self._n_total_samples
                )

                # Dispatch recherches locales
                if len(cand_y) > 0:
                    idx_cand_sorted = np.argsort(cand_y)

                    base_dispatch = 25 if self.dim > 15 else 20
                    decay = 0.97 ** iteration
                    n_dispatch = min(len(cand_y), max(5, int(base_dispatch * decay)))

                    futures = {}
                    for k in range(n_dispatch):
                        if self._should_stop():
                            break

                        idx = idx_cand_sorted[k]
                        x_start = cand_x[idx]

                        # Budget adaptatif
                        budget = min(
                            int(self.config.local_search_budget * (1.0 + 0.3 * (1.0 - k / n_dispatch))),
                            self.config.max_feval - self.n_evals
                        )

                        if budget < 100:
                            break

                        # Soumettre recherche locale
                        searcher = NUnirSearcher(self.objective, self.bounds,
                                                 self.config, self._stop_event)
                        future = self._executor.submit(searcher.search, x_start, budget)
                        futures[future] = x_start

                    # Collecter résultats
                    for future in as_completed(futures):
                        if self._should_stop():
                            break
                        try: 
                            x_opt, f_opt, n_ev = future.result(timeout=60)
                            self._increment_evals(n_ev)
                            self.stats['local_searches'] += 1

                            # Enregistrement du cluster
                            self.clusterer.add_cluster(x_opt, f_opt)

                            # Mise à jour du meilleur
                            if f_opt < best_ever_y: 
                                best_ever_y = f_opt
                                best_ever_x = x_opt.copy()
                                stagnation_counter = 0

                                self._best_ever = Sample(x=x_opt.copy(), y=f_opt, generation=iteration)

                                if callback:
                                    callback(self._best_ever)
                        except Exception: 
                            pass

                # Détection stagnation
                rel_improvement = abs(best_ever_y - last_best_y) / max(abs(best_ever_y), 1e-10)
                if rel_improvement < 1e-8:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                last_best_y = best_ever_y

                # Restart partiel si stagnation prolongée
                if stagnation_counter > 20 and self.clusterer.n_clusters > 5:
                    n_keep_clusters = max(3, self.clusterer.n_clusters // 3)
                    with self.clusterer._lock:
                        if len(self.clusterer.y_seeds_cache) > n_keep_clusters: 
                            sorted_idx = np.argsort(self.clusterer.y_seeds_cache)[:n_keep_clusters]
                            self.clusterer.x_seeds_cache = self.clusterer.x_seeds_cache[sorted_idx].copy()
                            self.clusterer.y_seeds_cache = self.clusterer.y_seeds_cache[sorted_idx].copy()
                            self.clusterer.clusters = [self.clusterer.clusters[i] for i in sorted_idx]
                    stagnation_counter = 0

                self.stats['clusters'] = self.clusterer.n_clusters

        finally:
            if self._executor:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None

        # Retour du meilleur résultat
        if best_ever_x is not None: 
            return Sample(x=best_ever_x, y=best_ever_y, generation=max_iter)

        best_cluster = self.clusterer.get_best_minimum()
        if best_cluster: 
            return Sample(x=best_cluster[0], y=best_cluster[1], generation=max_iter)

        return None

    def get_statistics(self) -> Dict[str, Any]: 
        """Retourne les statistiques d'optimisation"""
        return {
            **self.stats,
            'n_evals': self.n_evals,
            'best_value': self._best_ever.y if self._best_ever else None,
            'dimension': self.dim
        }

    def cleanup(self):
        """Nettoie les ressources"""
        self._all_samples.clear()
        self.clusterer.clear()
        self._best_ever = None
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None


# =============================================================================
# OPTIMIZATION ENGINE WITH PGLOBAL
# =============================================================================

def run_optimization_pglobal(target_data: pd.DataFrame, data_type: DataType, substrate: str,
                             thickness_min: float, thickness_max: float, lambda_min: float,
                             lambda_max: float, use_normalized:  bool, weight_T: float,
                             weight_R: float, is_frosted_glass: bool,
                             exclude_min:  Optional[float] = None, exclude_max: Optional[float] = None,
                             progress_callback=None) -> Dict[str, Any]: 
    """Exécute l'optimisation complète avec PGLOBAL."""

    # Filtrer les données
    mask = ((target_data['lambda'] >= lambda_min) & (target_data['lambda'] <= lambda_max))
    wls = target_data.loc[mask, 'lambda'].to_numpy()

    target_T = None
    target_R = None

    if data_type in (DataType.TRANSMISSION, DataType.BOTH) and not is_frosted_glass: 
        if 'T' in target_data.columns:
            target_T = target_data.loc[mask, 'T'].to_numpy()

    if data_type in (DataType.REFLECTION, DataType.BOTH):
        if 'R' in target_data.columns:
            target_R = target_data.loc[mask, 'R'].to_numpy()

    # Indices du substrat
    if is_frosted_glass:
        n_sub = get_n_frosted_glass_array(wls)
    else:
        sub_id = SUBSTRATES[substrate]["id"]
        n_sub = get_n_substrate_array_by_id(sub_id, wls)

    # Filtrer valeurs valides
    valid = np.isfinite(n_sub)
    if target_T is not None:
        valid &= np.isfinite(target_T)
    if target_R is not None:
        valid &= np.isfinite(target_R)

    wls = wls[valid]
    n_sub = n_sub[valid]
    if target_T is not None:
        target_T = target_T[valid]
    if target_R is not None: 
        target_R = target_R[valid]

    exclude_range = None
    if exclude_min is not None and exclude_max is not None:
        exclude_range = (exclude_min, exclude_max)

    # Créer l'objectif
    obj = TLUObjective(
        wls, target_T, target_R, n_sub, data_type,
        (thickness_min, thickness_max),
        use_normalized=use_normalized,
        weight_T=weight_T,
        weight_R=weight_R,
        exclude_range=exclude_range,
        is_frosted_glass=is_frosted_glass
    )

    best_mse = np.inf
    best_params = None

    if progress_callback: 
        progress_callback(5, "🚀 PGLOBAL Global Search...")

    # === Phase 1: PGLOBAL ===
    pg_conf = PGlobalConfig(
        max_feval=80000,
        max_time=180.0,
        n_samples_per_iter=8000,
        local_search_budget=3000,
        alpha=0.02,
        reduction_ratio=0.08
    )

    stop_event = Event()

    optimizer = PGlobalOptimizer(
        obj, obj.get_bounds(),
        config=pg_conf,
        log_indices=[2, 5],  # A et Eu en échelle log
        stop_event=stop_event,
        n_workers=max(1, (os.cpu_count() or 4) - 1)
    )

    def pglobal_callback(sample: Sample):
        if sample.y < best_mse: 
            progress = min(45, int(45 * optimizer.n_evals / 80000))
            if progress_callback:
                progress_callback(progress, f"PGLOBAL: MSE={sample.y:.2e}")

    best_sample = optimizer.optimize(max_iter=30, callback=pglobal_callback)

    if best_sample: 
        best_params = best_sample.x.copy()
        best_mse = best_sample.y
    else:
        best_params = np.array([
            (thickness_min + thickness_max) / 2,
            2.5, 100.0, 4.0, 1.0, 0.1, 2.5
        ])

    n_evals_pglobal = optimizer.n_evals
    optimizer.cleanup()

    if progress_callback: 
        progress_callback(50, "🔧 L-BFGS-B Polish...")

    # === Phase 2: L-BFGS-B Polish ===
    bounds = obj.get_bounds()
    res = scipy.optimize.minimize(
        obj, best_params, method='L-BFGS-B',
        bounds=[(lb, ub) for lb, ub in bounds],
        options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 2000}
    )

    if res.fun < best_mse:
        best_mse = res.fun
        best_params = res.x

    if progress_callback: 
        progress_callback(70, "🎯 Deep Coordinate Descent...")

    # === Phase 3:  Coordinate Descent ===
    weights = obj.weights
    use_T = data_type in (DataType.TRANSMISSION, DataType.BOTH) and not is_frosted_glass
    use_R = data_type in (DataType.REFLECTION, DataType.BOTH)

    tgt_T = target_T if target_T is not None else np.zeros_like(wls)
    tgt_R = target_R if target_R is not None else np.zeros_like(wls)

    for i in range(3):
        params_ref, mse_ref = run_coordinate_descent_RT(
            best_params, wls, tgt_T, tgt_R, n_sub, weights,
            use_T, use_R, use_normalized, weight_T, weight_R,
            is_frosted_glass, max_iter=1000
        )
        if mse_ref < best_mse:
            best_mse = mse_ref
            best_params = params_ref
        if progress_callback: 
            progress_callback(75 + i * 8, f"Refine #{i+1}:  MSE={best_mse:.2e}")

    if progress_callback: 
        progress_callback(100, "✅ Done!")

    # Package results
    thickness = best_params[0]
    tlu_params = TLUParameters.from_array(best_params[1:7])

    # Calcul des résultats sur tout le spectre
    l_full = target_data['lambda'].to_numpy()
    E_full = HC_EV_NM / l_full

    if is_frosted_glass:
        n_sub_full = get_n_frosted_glass_array(l_full)
    else: 
        sub_id = SUBSTRATES[substrate]["id"]
        n_sub_full = get_n_substrate_array_by_id(sub_id, l_full)

    eps2 = epsilon2_TLU_array(E_full, tlu_params.Eg, tlu_params.A,
                              tlu_params.E0, tlu_params.C, tlu_params.Eu)
    eps1 = epsilon1_TL_analytic(E_full, tlu_params.Eg, tlu_params.A,
                                tlu_params.E0, tlu_params.C, tlu_params.eps_inf)
    n_calc, k_calc, _ = epsilon_to_nk(eps1, eps2, 0.5, 15.0, 15.0)

    if is_frosted_glass:
        T_calc = np.full_like(l_full, np.nan)
        T_sub = np.full_like(l_full, np.nan)
        R_calc = calculate_reflection_infinite_substrate_array(l_full, n_calc, k_calc, thickness, n_sub_full)
        R_sub = calculate_R_frosted_glass_reference(l_full, n_sub_full)
    else:
        T_calc = calculate_transmission_array(l_full, n_calc, k_calc, thickness, n_sub_full)
        T_sub = calculate_T_substrate_array(l_full, n_sub_full)
        R_calc = calculate_reflection_array(l_full, n_calc, k_calc, thickness, n_sub_full)
        R_sub = calculate_R_substrate_array(l_full, n_sub_full)

    with np.errstate(divide='ignore', invalid='ignore'):
        T_sub_safe = np.where(T_sub > SMALL_EPSILON, T_sub, 1.0)
        T_norm_calc = np.where(T_sub > SMALL_EPSILON, T_calc / T_sub_safe, np.nan)

        R_sub_safe = np.where(R_sub > SMALL_EPSILON, R_sub, 1.0)
        R_norm_calc = np.where(R_sub > SMALL_EPSILON, R_calc / R_sub_safe, np.nan)

    df_results = pd.DataFrame({
        'lambda':  l_full,
        'n_calc': n_calc,
        'k_calc': k_calc,
        'alpha_cm-1': 4.0 * PI * k_calc / (l_full * 1e-7),
        'T_calc': T_calc * 100,
        'T_norm_calc': T_norm_calc * 100,
        'R_calc': R_calc * 100,
        'R_norm_calc':  R_norm_calc * 100,
    })

    if 'T' in target_data.columns:
        df_results['T_target'] = target_data['T'].to_numpy()
    if 'R' in target_data.columns:
        df_results['R_target'] = target_data['R'].to_numpy()

    return {
        'thickness': thickness,
        'mse': best_mse,
        'tlu_params': tlu_params,
        'df_results': df_results,
        'n_evals': obj.n_evals + n_evals_pglobal,
        'optimization_method': 'PGLOBAL + L-BFGS-B + Coordinate Descent'
    }


# =============================================================================
# STREAMLIT APP
# =============================================================================

# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    # 1.Configuration de la page avec l'icône certus.ico
    st.set_page_config(
        page_title="CERTUS-INDEX PGLOBAL",
        page_icon="certus.ico" if os.path.exists("certus.ico") else "🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 2.Styles CSS personnalisés
    st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background:  linear-gradient(90deg, #05103b, #0ea5e9);
        color: white;
        font-weight: bold;
        border:  none;
        padding: 0.75rem;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background:  linear-gradient(90deg, #1e40af, #0284c7);
    }
    .info-box {
        background: #f1f5f9;
        border-left: 4px solid #1e3a8a;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background: #ecfdf5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        margin:  1rem 0;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    # 3.Affichage du logo SVG
    st.markdown(LOGO_SVG, unsafe_allow_html=True)

    # 4.Initialisation de l'état de la session
    if 'target_data' not in st.session_state: 
        st.session_state.target_data = None
    if 'data_type' not in st.session_state: 
        st.session_state.data_type = None
    if 'results' not in st.session_state: 
        st.session_state.results = None
    if 'source_filename' not in st.session_state: 
        st.session_state.source_filename = ""

    if 'config' not in st.session_state: 
        st.session_state.config = {
            'substrate':  'SiO2',
            'is_frosted_glass': False,
            'thickness_min': 50.0,
            'thickness_max': 1000.0,
            'lambda_min': 300.0,
            'lambda_max':  900.0,
            'use_normalized':  True,
            'weight_T': 1.0,
            'weight_R': 1.0,
            'use_exclusion': False,
            'exclude_min': 400.0,
            'exclude_max':  450.0,
        }

    # ==========================================
    # SIDEBAR :  Chargement et Configuration
    # ==========================================
    with st.sidebar:
        st.header("📁 Données d'entrée")
        uploaded_file = st.file_uploader(
            "Charger un fichier de spectre",
            type=['csv', 'txt', 'xlsx', 'xls']
        )

        if uploaded_file is not None:  
            try:
                if uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file, header=0)
                else:
                    uploaded_file.seek(0)
                    # LIRE LES PREMIÈRES LIGNES POUR DÉTECTER LE SÉPARATEUR
                    content = uploaded_file.read(2048).decode('utf-8')
                    uploaded_file.seek(0)
                    
                    # Priorité au point-virgule si présent (format européen classique)
                    if ';' in content:
                        detected_sep = ';'
                    else:
                        detected_sep = None  # Laisser pandas deviner pour les autres cas

                    df = pd.read_csv(uploaded_file, sep=detected_sep, engine='python', header=None)

                    # Nettoyage robuste des décimales (remplace , par . et convertit en numérique)
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            df[col] = df[col].str.replace(',', '. ', regex=False)
                    
                    df = df.apply(pd.to_numeric, errors='coerce')

                    # Gestion du header :  si la première ligne est devenue NaN après conversion, 
                    # c'était probablement des titres (Wavelength, etc.), on les supprime. 
                    if df.iloc[0].isna().any():
                        df = df.iloc[1:].reset_index(drop=True)
                    
                    # Suppression des lignes vides restantes
                    df = df. dropna().reset_index(drop=True)

                # Suite du traitement (commun CSV et Excel)
                n_cols = min(3, len(df.columns))
                df = df.iloc[:, :n_cols]
                df = df.sort_values(by=df.columns[0]).reset_index(drop=True)

                

                data_type, parsed_data = analyze_loaded_data(df)

                # Normalisation automatique (0-1)
                if parsed_data['T'] is not None and len(parsed_data['T']) > 0 and np.nanmax(parsed_data['T']) > 1.5:
                    parsed_data['T'] = parsed_data['T'] / 100.0
                if parsed_data['R'] is not None and len(parsed_data['R']) > 0 and np.nanmax(parsed_data['R']) > 1.5:
                    parsed_data['R'] = parsed_data['R'] / 100.0

                target_data = pd.DataFrame({'lambda': parsed_data['lambda']})
                if parsed_data['T'] is not None: 
                    target_data['T'] = parsed_data['T']
                if parsed_data['R'] is not None: 
                    target_data['R'] = parsed_data['R']

                st.session_state.target_data = target_data
                st.session_state.data_type = data_type
                st.session_state.source_filename = uploaded_file.name
                st.session_state.config['lambda_min'] = float(target_data['lambda'].min())
                st.session_state.config['lambda_max'] = float(target_data['lambda'].max())

                st.success(f"✅ Chargé :  {uploaded_file.name}")

                # Affichage du type détecté
                type_labels = {
                    DataType.TRANSMISSION:  "📊 Détecté :  TRANSMISSION uniquement",
                    DataType.REFLECTION: "📊 Détecté :  RÉFLEXION uniquement",
                    DataType.BOTH: "📊 Détecté :  TRANSMISSION + RÉFLEXION"
                }
                st.info(type_labels.get(data_type, "📊 Type détecté"))

            except Exception as e:
                st.error(f"❌ Erreur de chargement : {str(e)}")

        st.divider()
        st.header("🧱 Substrat")
        substrate_mode = st.radio("Mode Substrat", options=["Standard", "Frosted Glass"])
        st.session_state.config['is_frosted_glass'] = (substrate_mode == "Frosted Glass")

        if not st.session_state.config['is_frosted_glass']: 
            st.session_state.config['substrate'] = st.selectbox(
                "Matériau",
                options=SUBSTRATE_LIST,
                index=SUBSTRATE_LIST.index(st.session_state.config['substrate']) if st.session_state.config['substrate'] in SUBSTRATE_LIST else 0
            )
        else:
            st.session_state.config['substrate'] = "SiO2"
            st.info("ℹ️ Mode Frosted Glass :  Réflexion uniquement, substrat infini")

        st.divider()
        st.header("⚙️ Configuration")

        c1, c2 = st.columns(2)
        with c1:
            st.session_state.config['thickness_min'] = st.number_input(
                "Épaisseur Min (nm)",
                value=st.session_state.config['thickness_min'],
                min_value=10.0,
                max_value=5000.0
            )
        with c2:
            st.session_state.config['thickness_max'] = st.number_input(
                "Épaisseur Max (nm)",
                value=st.session_state.config['thickness_max'],
                min_value=10.0,
                max_value=5000.0
            )

        l1, l2 = st.columns(2)
        with l1:
            st.session_state.config['lambda_min'] = st.number_input(
                "λ Min (nm)",
                value=st.session_state.config['lambda_min'],
                min_value=190.0,
                max_value=3500.0
            )
        with l2:
            st.session_state.config['lambda_max'] = st.number_input(
                "λ Max (nm)",
                value=st.session_state.config['lambda_max'],
                min_value=190.0,
                max_value=3500.0
            )

        # Normalisation
        st.session_state.config['use_normalized'] = st.checkbox(
            "Utiliser données normalisées (T/T_sub, R/R_sub)",
            value=st.session_state.config['use_normalized'] if not st.session_state.config['is_frosted_glass'] else False,
            disabled=st.session_state.config['is_frosted_glass']
        )
        if st.session_state.config['is_frosted_glass']:
            st.session_state.config['use_normalized'] = False

        # Poids
        st.subheader("Poids")
        w1, w2 = st.columns(2)
        with w1:
            st.session_state.config['weight_T'] = st.number_input(
                "Poids T",
                value=0.0 if st.session_state.config['is_frosted_glass'] else st.session_state.config['weight_T'],
                min_value=0.0,
                max_value=10.0,
                disabled=st.session_state.config['is_frosted_glass']
            )
        with w2:
            st.session_state.config['weight_R'] = st.number_input(
                "Poids R",
                value=st.session_state.config['weight_R'],
                min_value=0.0,
                max_value=10.0
            )

        # Zone d'exclusion
        with st.expander("🚫 Zone d'exclusion", expanded=False):
            st.session_state.config['use_exclusion'] = st.checkbox(
                "Activer la zone d'exclusion",
                value=st.session_state.config['use_exclusion']
            )
            if st.session_state.config['use_exclusion']: 
                ex1, ex2 = st.columns(2)
                with ex1:
                    st.session_state.config['exclude_min'] = st.number_input(
                        "Exclusion Min (nm)",
                        value=st.session_state.config['exclude_min']
                    )
                with ex2:
                    st.session_state.config['exclude_max'] = st.number_input(
                        "Exclusion Max (nm)",
                        value=st.session_state.config['exclude_max']
                    )

        st.divider()

        # Bouton d'optimisation
        run_disabled = st.session_state.target_data is None

        if st.button("🚀 LANCER OPTIMISATION PGLOBAL", disabled=run_disabled, type="primary"):
            if st.session_state.target_data is not None:
                cfg = st.session_state.config

                # Vérification pour frosted glass
                if cfg['is_frosted_glass'] and 'R' not in st.session_state.target_data.columns:
                    st.error("❌ Le mode Frosted Glass nécessite des données de Réflexion (R) !")
                else:
                    effective_data_type = DataType.REFLECTION if cfg['is_frosted_glass'] else st.session_state.data_type

                    with st.spinner("🔄 Optimisation PGLOBAL en cours..  Cela peut prendre quelques minute. "):
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        def progress_callback(pct, msg):
                            progress_bar.progress(pct / 100)
                            status_text.text(msg)

                        try:
                            results = run_optimization_pglobal(
                                target_data=st.session_state.target_data,
                                data_type=effective_data_type,
                                substrate=cfg['substrate'],
                                thickness_min=cfg['thickness_min'],
                                thickness_max=cfg['thickness_max'],
                                lambda_min=cfg['lambda_min'],
                                lambda_max=cfg['lambda_max'],
                                use_normalized=cfg['use_normalized'],
                                weight_T=cfg['weight_T'],
                                weight_R=cfg['weight_R'],
                                is_frosted_glass=cfg['is_frosted_glass'],
                                exclude_min=cfg['exclude_min'] if cfg['use_exclusion'] else None,
                                exclude_max=cfg['exclude_max'] if cfg['use_exclusion'] else None,
                                progress_callback=progress_callback
                            )

                            st.session_state.results = results
                            st.session_state.results['substrate'] = cfg['substrate']
                            st.session_state.results['is_frosted_glass'] = cfg['is_frosted_glass']
                            st.session_state.results['use_normalized'] = cfg['use_normalized']
                            st.session_state.results['data_type'] = effective_data_type
                            st.session_state.results['lambda_min'] = cfg['lambda_min']
                            st.session_state.results['lambda_max'] = cfg['lambda_max']

                            progress_bar.progress(100)
                            status_text.text("✅ Optimisation terminée !")
                            st.rerun()

                        except Exception as e: 
                            st.error(f"❌ Échec de l'optimisation : {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())

    # ==========================================
    # CONTENU PRINCIPAL
    # ==========================================
    if st.session_state.target_data is not None:
        cfg = st.session_state.config

        # Création des onglets
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Spectre", "🔢 n & k", "📊 Tableau de données", "💡 À propos"])

        with tab1:
            st.subheader("Spectre de Transmission / Réflexion")

            fig = go.Figure()
            target_data = st.session_state.target_data
            wls = target_data['lambda'].values

            # Tracer les données cibles
            if 'T' in target_data.columns:
                fig.add_trace(go.Scatter(
                    x=wls,
                    y=target_data['T'].values * 100,
                    mode='markers',
                    name='T Cible',
                    marker=dict(size=4, color='#10b981'),
                ))

            if 'R' in target_data.columns:
                fig.add_trace(go.Scatter(
                    x=wls,
                    y=target_data['R'].values * 100,
                    mode='markers',
                    name='R Cible',
                    marker=dict(size=4, color='#ef4444'),
                ))

            # Tracer l'ajustement si disponible
            if st.session_state.results is not None: 
                results = st.session_state.results
                df_res = results['df_results']
                use_norm = results.get('use_normalized', False)
                is_fg = results.get('is_frosted_glass', False)
                res_lambda_min = results.get('lambda_min', cfg['lambda_min'])
                res_lambda_max = results.get('lambda_max', cfg['lambda_max'])

                mask = (df_res['lambda'] >= res_lambda_min) & (df_res['lambda'] <= res_lambda_max)
                df_plot = df_res[mask]

                if not is_fg and results.get('data_type') in (DataType.TRANSMISSION, DataType.BOTH):
                    col_T = 'T_norm_calc' if use_norm else 'T_calc'
                    if col_T in df_plot.columns:
                        fig.add_trace(go.Scatter(
                            x=df_plot['lambda'].values,
                            y=df_plot[col_T].values,
                            mode='lines',
                            name='T Ajusté',
                            line=dict(width=3, color='#059669'),
                        ))

                if results.get('data_type') in (DataType.REFLECTION, DataType.BOTH) or is_fg: 
                    col_R = 'R_norm_calc' if use_norm else 'R_calc'
                    if col_R in df_plot.columns:
                        fig.add_trace(go.Scatter(
                            x=df_plot['lambda'].values,
                            y=df_plot[col_R].values,
                            mode='lines',
                            name='R Ajusté',
                            line=dict(width=3, color='#dc2626'),
                        ))

            fig.update_layout(
                xaxis_title="Longueur d'onde (nm)",
                yaxis_title="T/R (%)",
                hovermode='x unified',
                template='plotly_white',
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Constantes Optiques (n, k)")

            if st.session_state.results is not None: 
                results = st.session_state.results
                df_res = results['df_results']
                res_lambda_min = results.get('lambda_min', cfg['lambda_min'])
                res_lambda_max = results.get('lambda_max', cfg['lambda_max'])

                mask = (df_res['lambda'] >= res_lambda_min) & (df_res['lambda'] <= res_lambda_max)
                df_plot = df_res[mask]

                fig = make_subplots(specs=[[{"secondary_y": True}]])

                fig.add_trace(
                    go.Scatter(
                        x=df_plot['lambda'].values,
                        y=df_plot['n_calc'].values,
                        name="n",
                        line=dict(width=3, color='#1e3a8a')
                    ),
                    secondary_y=False,
                )

                fig.add_trace(
                    go.Scatter(
                        x=df_plot['lambda'].values,
                        y=df_plot['k_calc'].values,
                        name="k",
                        line=dict(width=3, color='#f59e0b')
                    ),
                    secondary_y=True,
                )

                fig.update_xaxes(title_text="Longueur d'onde (nm)")
                fig.update_yaxes(title_text="Indice de réfraction (n)", secondary_y=False)
                fig.update_yaxes(title_text="Coefficient d'extinction (k)", secondary_y=True)

                fig.update_layout(
                    hovermode='x unified',
                    template='plotly_white',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02),
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                st.divider()

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        label="Épaisseur",
                        value=f"{results['thickness']:.2f} nm"
                    )

                with col2:
                    st.metric(
                        label="MSE",
                        value=f"{results['mse']:.2e}"
                    )

                with col3:
                    st.metric(
                        label="Bandgap (Eg)",
                        value=f"{results['tlu_params'].Eg:.3f} eV"
                    )

                with col4:
                    st.metric(
                        label="ε∞",
                        value=f"{results['tlu_params'].eps_inf:.3f}"
                    )

                
                # Correction de la ligne 1534
                st.markdown(f"""
                <div class="success-box">
                    <strong>🎯 Méthode d'optimisation :</strong> {results.get('optimization_method', 'N/A')}<br>
                    <strong>📊 Évaluations totales :</strong> {results.get('n_evals', 'N/A'):,}
                </div>
                """, unsafe_allow_html=True)
                with st.expander("📐 Paramètres TLU complets", expanded=False):
                    tlu = results['tlu_params']
                    params_df = pd.DataFrame({
                        'Paramètre': ['Eg (eV)', 'A (eV)', 'E0 (eV)', 'C (eV)', 'Eu (eV)', 'ε∞'],
                        'Valeur': [f"{tlu.Eg:.4f}", f"{tlu.A:.4f}", f"{tlu.E0:.4f}",
                                   f"{tlu.C:.4f}", f"{tlu.Eu:.4f}", f"{tlu.eps_inf:.4f}"],
                        'Description': [
                            'Énergie de bande interdite',
                            'Amplitude',
                            'Énergie de pic',
                            'Élargissement',
                            'Énergie d\'Urbach',
                            'Constante diélectrique haute fréquence'
                        ]
                    })
                    st.dataframe(params_df, use_container_width=True, hide_index=True)
            else:
                st.info("👆 Lancez l'optimisation pour voir les résultats n & k")

        with tab3:
            st.subheader("Tableau des résultats")

            if st.session_state.results is not None:
                results = st.session_state.results
                df_res = results['df_results']
                res_lambda_min = results.get('lambda_min', cfg['lambda_min'])
                res_lambda_max = results.get('lambda_max', cfg['lambda_max'])

                mask = (df_res['lambda'] >= res_lambda_min) & (df_res['lambda'] <= res_lambda_max)
                df_display = df_res[mask].copy()

                display_cols = ['lambda', 'n_calc', 'k_calc']
                if not results.get('is_frosted_glass', False):
                    display_cols.extend(['T_calc', 'R_calc'])
                else:
                    display_cols.append('R_calc')

                if 'T_target' in df_display.columns:
                    display_cols.append('T_target')
                if 'R_target' in df_display.columns:
                    display_cols.append('R_target')

                df_show = df_display[display_cols].round(4)

                st.dataframe(df_show, use_container_width=True, height=400)

                st.divider()

                col1, col2 = st.columns(2)

                with col1:
                    csv_buffer = io.StringIO()
                    df_res.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()

                    st.download_button(
                        label="📥 Télécharger CSV",
                        data=csv_data,
                        file_name=f"certus_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )

                with col2:
                    excel_buffer = io.BytesIO()

                    tlu = results['tlu_params']
                    summary_data = {
                        "Propriété": [
                            "Fichier source", "Date", "Version logiciel", "Substrat",
                            "Mode substrat", "Méthode d'optimisation", "MSE final",
                            "Évaluations totales", "Épaisseur optimale (nm)",
                            "Bandgap Eg (eV)", "Amplitude A (eV)", "Énergie pic E0 (eV)",
                            "Élargissement C (eV)", "Énergie Urbach Eu (eV)", "Epsilon infini"
                        ],
                        "Valeur": [
                            st.session_state.source_filename,
                            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "CERTUS-INDEX Streamlit v5.0 (PGLOBAL)",
                            results.get('substrate', 'N/A'),
                            "Frosted Glass" if results.get('is_frosted_glass') else "Standard",
                            results.get('optimization_method', 'N/A'),
                            f"{results['mse']:.6e}",
                            f"{results.get('n_evals', 0):,}",
                            f"{results['thickness']:.2f}",
                            f"{tlu.Eg:.4f}",
                            f"{tlu.A:.4f}",
                            f"{tlu.E0:.4f}",
                            f"{tlu.C:.4f}",
                            f"{tlu.Eu:.4f}",
                            f"{tlu.eps_inf:.4f}"
                        ]
                    }
                    df_summary = pd.DataFrame(summary_data)

                    try:
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            df_summary.to_excel(writer, sheet_name='Résumé', index=False)
                            df_res[['lambda', 'n_calc', 'k_calc', 'alpha_cm-1']].to_excel(
                                writer, sheet_name='Constantes Optiques', index=False
                            )
                            df_res.to_excel(writer, sheet_name='Données Complètes', index=False)

                        excel_data = excel_buffer.getvalue()

                        st.download_button(
                            label="📥 Télécharger Rapport Excel",
                            data=excel_data,
                            file_name=f"certus_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except ImportError:
                        st.warning("⚠️ openpyxl non installé.Installez avec :  pip install openpyxl")
            else:
                st.info("👆 Lancez l'optimisation pour voir le tableau de données")

        with tab4:
            st.subheader("À propos de CERTUS-INDEX Édition PGLOBAL")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                ### 🚀 Moteur d'optimisation PGLOBAL
                **Algorithme d'optimisation globale hybride** combinant : 
                - **Échantillonnage aléatoire** avec densité adaptative
                - **Clustering single-linkage** pour éviter les recherches redondantes
                - **Recherche locale N-UNIR** avec accélération de ligne
                - **L-BFGS-B** polissage quasi-Newton
                - **Descente de coordonnées** raffinement profond

                Cela garantit de trouver le **minimum global** plutôt que de rester bloqué dans des minima locaux.
                """)

            with col2:
                st.markdown("""
                ### ⚡ Caractéristiques de performance
                - **Compilation JIT** via Numba pour une vitesse proche du C++
                - **Évaluation parallèle** avec ThreadPoolExecutor
                - **Allocation de budget adaptative** pour les recherches locales
                - **Détection de stagnation** avec redémarrage automatique
                - **Échantillonnage en échelle log** pour les paramètres d'amplitude
                """)

            st.divider()

            st.markdown("""
            ### 📚 Modèle Tauc-Lorentz-Urbach

            Le modèle TLU combine :
            - **Modèle de Tauc** pour l'absorption de bande interdite
            - **Oscillateur de Lorentz** pour les transitions interbandes
            - **Queue d'Urbach** pour l'absorption sous-bande

            #### Paramètres : 
            | Paramètre | Description | Plage typique |
            |-----------|-------------|---------------|
            | **Eg** | Énergie de bande interdite | 0.5 - 6.0 eV |
            | **A** | Amplitude de l'oscillateur | 10 - 2000 eV |
            | **E0** | Énergie de transition pic | 1.5 - 10 eV |
            | **C** | Paramètre d'élargissement | 0.1 - 10 eV |
            | **Eu** | Énergie d'Urbach | 0.01 - 3 eV |
            | **ε∞** | Constante diélectrique haute fréquence | 1 - 10 |

            #### Cohérence Kramers-Kronig
            Le modèle assure la cohérence physique entre n et k via
            les relations analytiques de Kramers-Kronig.
            """)

            st.divider()

            st.markdown("""
            ### 🔄 Pipeline d'optimisation

            ```
            ┌─────────────────────────────────────────────────────────────┐
            │  Phase 1 :  Recherche globale PGLOBAL (~80 000 évaluations)  │
            │  ├─ Échantillonnage aléatoire adaptatif                     │
            │  ├─ Clustering single-linkage                               │
            │  └─ Recherches locales N-UNIR avec accélération de ligne    │
            ├─────────────────────────────────────────────────────────────┤
            │  Phase 2 :  Polissage L-BFGS-B (~2 000 itérations)           │
            │  └─ Descente de gradient quasi-Newton                       │
            ├─────────────────────────────────────────────────────────────┤
            │  Phase 3 : Raffinement par descente de coordonnées          │
            │            (3×1000 itérations)                              │
            │  └─ Optimisation paramètre par paramètre                    │
            └─────────────────────────────────────────────────────────────┘
            ```
            """)

            st.divider()

            st.markdown("""
            ### 📖 Comment utiliser

            1.**Chargez** votre fichier de spectre (CSV, TXT ou Excel)
            2.**Configurez** le type de substrat et les paramètres d'optimisation
            3.**Lancez** l'optimisation PGLOBAL
            4.**Exportez** les résultats en CSV ou rapport Excel

            #### Formats de fichiers supportés : 
            - Colonnes : `longueur_d'onde, T` ou `longueur_d'onde, R` ou `longueur_d'onde, T, R`
            - Les valeurs peuvent être en % (0-100) ou en fraction (0-1)
            - Détection automatique du type de données
            """)

    else:
        # Écran d'accueil quand aucune donnée n'est chargée
        st.markdown("""
        <div class="info-box">
        <h3>👋 Bienvenue dans CERTUS-INDEX Édition PGLOBAL</h3>
        <p>Programmé par Fabien Lemarchand.</p>
        <p>Chargez un fichier de spectre dans la barre latérale pour commencer.</p>
        <p>Formats supportés : CSV, TXT, Excel (XLSX/XLS)</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            ### 🔬 Constantes Optiques
            Extrayez l'indice de réfraction (n) et le coefficient d'extinction (k)
            à partir de spectres de transmission et/ou réflexion.
            """)

        with col2:
            st.markdown("""
            ### 🚀 Moteur PGLOBAL
            Optimisation globale avancée avec clustering
            pour trouver le véritable minimum global.
            """)

        with col3:
            st.markdown("""
            ### ⚡ Rapide & Précis
            Moteur physique compilé JIT avec optimisation
            parallèle pour des performances maximales.
            """)





# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__": 
    main()
