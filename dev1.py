# -*- coding: utf-8 -*- # Nécessaire pour les caractères spéciaux dans les commentaires/strings
import logging
import streamlit as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.interpolate import CubicSpline
import os
import time
import multiprocessing
import numba
import traceback
import pandas as pd
import warnings
import math
import io

# --- Configuration du Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constantes Globales ---
SMALL_EPSILON = 1e-9
PI = np.pi
HUGE_PENALTY = 1e30
N_KNOT_VALUE_BOUNDS = (1.2, 4.0)
LOG_K_KNOT_VALUE_BOUNDS = (math.log(1e-6), math.log(5.0))
SUBSTRATE_LIST = ["SiO2", "N-BK7", "D263T eco", "Sapphire", "B270i"]
SUBSTRATE_MIN_LAMBDA = {
    "SiO2": 230.0, "N-BK7": 400.0, "D263T eco": 360.0,
    "Sapphire": 230.0, "B270i": 400.0,
}

# --- Fonctions Utilitaires ---
def get_substrate_min_lambda(substrate_name):
    return SUBSTRATE_MIN_LAMBDA.get(substrate_name, 200.0)

# --- Fonctions de Calcul Optique (Numba JIT) ---
@numba.jit(nopython=True, cache=False)
def sellmeier_calc(l_um_sq, B1, C1, B2, C2, B3, C3):
    n_sq_minus_1 = (B1 * l_um_sq / (l_um_sq - C1)) + \
                   (B2 * l_um_sq / (l_um_sq - C2)) + \
                   (B3 * l_um_sq / (l_um_sq - C3))
    n_sq = n_sq_minus_1 + 1
    if n_sq < 1e-6: n_sq = 1e-6
    return np.sqrt(n_sq)

@numba.jit(nopython=True, cache=False)
def get_n_substrate(substrate_id, wavelength_nm):
    l_um = wavelength_nm / 1000.0
    l_um_sq = l_um * l_um
    # ... (le reste de la fonction get_n_substrate est inchangé)
    if substrate_id == 0: # SiO2
        min_wl = 230.0
        if wavelength_nm < min_wl: return np.nan
        B1=0.6961663; C1=0.0684043**2; B2=0.4079426; C2=0.1162414**2; B3=0.8974794; C3=9.896161**2
        return sellmeier_calc(l_um_sq, B1, C1, B2, C2, B3, C3)
    elif substrate_id == 1: # N-BK7
        min_wl = 400.0
        if wavelength_nm < min_wl: return np.nan
        B1=1.03961212; C1=0.00600069867; B2=0.231792344; C2=0.0200179144; B3=1.01046945; C3=103.560653
        return sellmeier_calc(l_um_sq, B1, C1, B2, C2, B3, C3)
    elif substrate_id == 2: # D263T eco
        min_wl = 360.0
        if wavelength_nm < min_wl: return np.nan
        B1=0.90963095; C1=0.0047563071; B2=0.37290409; C2=0.01621977; B3=0.92110613; C3=105.77911
        return sellmeier_calc(l_um_sq, B1, C1, B2, C2, B3, C3)
    elif substrate_id == 3: # Sapphire
        min_wl = 230.0
        if wavelength_nm < min_wl: return np.nan
        # Coefficients pour Sapphire (ordinaire, source diverse, vérifier si adéquat)
        # Il manque des coefficients ici, j'utilise SiO2 comme fallback, à corriger si nécessaire!
        # B1 = 1.43134930; C1 = 0.0052799261; B2 = 0.65054713; C2 = 0.014238474; B3 = 5.3414021; C3 = 325.01783
        # Utilisation de SiO2 en attendant une correction
        B1=0.6961663; C1=0.0684043**2; B2=0.4079426; C2=0.1162414**2; B3=0.8974794; C3=9.896161**2
        return sellmeier_calc(l_um_sq, B1, C1, B2, C2, B3, C3)
    elif substrate_id == 4: # B270i
        min_wl = 400.0
        if wavelength_nm < min_wl: return np.nan
        B1=0.90110328; C1=0.0045578115; B2=0.39734436; C2=0.016601149; B3=0.94615601; C3=111.88593
        return sellmeier_calc(l_um_sq, B1, C1, B2, C2, B3, C3)
    else: # Fallback sur SiO2 si ID inconnu
        min_wl = 230.0
        if wavelength_nm < min_wl: return np.nan
        B1=0.6961663; C1=0.0684043**2; B2=0.4079426; C2=0.1162414**2; B3=0.8974794; C3=9.896161**2
        return sellmeier_calc(l_um_sq, B1, C1, B2, C2, B3, C3)


@numba.jit(nopython=True, cache=False)
def calculate_monolayer_lambda(l_val, nMono, thickness_nm, nSub_val):
    # ... (le reste de la fonction calculate_monolayer_lambda est inchangé)
    if not np.isfinite(nSub_val) or nSub_val < 1e-6 : return np.nan, np.nan, np.nan
    n0 = 1.0
    eta_inc_fwd = np.complex128(n0 + 0j); eta_sub_fwd = np.complex128(nSub_val + 0j)
    eta_inc_rev = eta_sub_fwd; eta_sub_rev = eta_inc_fwd
    real_eta_inc_fwd = n0; real_eta_sub_fwd = nSub_val
    real_eta_inc_rev = nSub_val; real_eta_sub_rev = n0

    # Interface Substrat -> Milieu Sortie (Air)
    Rb_plus = 0.0; Tb_plus = 1.0
    denom_b_fwd = eta_sub_fwd + eta_inc_fwd
    if abs(denom_b_fwd) > SMALL_EPSILON:
        rb_plus_amp = (eta_inc_fwd - eta_sub_fwd) / denom_b_fwd; Rb_plus = abs(rb_plus_amp)**2
        tb_plus_amp = (2.0 * eta_inc_fwd) / denom_b_fwd
        if abs(real_eta_inc_fwd) > SMALL_EPSILON: Tb_plus = (real_eta_sub_fwd / real_eta_inc_fwd) * (abs(tb_plus_amp)**2)
        else: Tb_plus = 0.0
    else: Rb_plus = 1.0; Tb_plus = 0.0

    Rb_minus = 0.0; Tb_minus = 1.0
    denom_b_rev = eta_sub_rev + eta_inc_rev
    if abs(denom_b_rev) > SMALL_EPSILON:
        rb_minus_amp = (eta_inc_rev - eta_sub_rev) / denom_b_rev; Rb_minus = abs(rb_minus_amp)**2
        tb_minus_amp = (2.0 * eta_inc_rev) / denom_b_rev
        if abs(real_eta_inc_rev) > SMALL_EPSILON: Tb_minus = (real_eta_sub_rev / real_eta_inc_rev) * (abs(tb_minus_amp)**2)
        else: Tb_minus = 0.0
    else: Rb_minus = 1.0; Tb_minus = 0.0

    # Matrice caractéristique de la monocouche
    eta1 = nMono; phi1 = (2.0 * PI / l_val) * eta1 * thickness_nm; cos_phi1 = np.cos(phi1); sin_phi1 = np.sin(phi1)
    M1 = np.zeros((2, 2), dtype=np.complex128); M1[0, 0] = cos_phi1; M1[1, 1] = cos_phi1
    if abs(eta1) > SMALL_EPSILON: M1[0, 1] = (1j / eta1) * sin_phi1
    else: M1[0, 1] = np.complex128(0 + 1j*HUGE_PENALTY) if abs(sin_phi1) > SMALL_EPSILON else 0j
    M1[1, 0] = 1j * eta1 * sin_phi1

    M_fwd = M1; M_rev = M1

    # Calcul des coefficients de réflexion/transmission de la structure complète
    B_fwd = M_fwd[0, 0]; C_fwd = M_fwd[0, 1]; A_fwd = M_fwd[1, 0]; D_fwd = M_fwd[1, 1]
    denom_a_plus = (eta_inc_fwd * B_fwd + eta_inc_fwd * eta_sub_fwd * C_fwd + A_fwd + eta_sub_fwd * D_fwd)
    if abs(denom_a_plus) < SMALL_EPSILON: denom_a_plus = np.complex128(SMALL_EPSILON + 0j)
    ra_plus_amp = (eta_inc_fwd * B_fwd + eta_inc_fwd * eta_sub_fwd * C_fwd - A_fwd - eta_sub_fwd * D_fwd) / denom_a_plus
    ta_plus_amp = (2.0 * eta_inc_fwd) / denom_a_plus
    Ra_plus = abs(ra_plus_amp)**2; Ta_plus = 0.0
    if abs(real_eta_inc_fwd) > SMALL_EPSILON: Ta_plus = (real_eta_sub_fwd / real_eta_inc_fwd) * abs(ta_plus_amp)**2

    B_rev = M_rev[0, 0]; C_rev = M_rev[0, 1]; A_rev = M_rev[1, 0]; D_rev = M_rev[1, 1]
    denom_a_minus = (eta_inc_rev * B_rev + eta_inc_rev * eta_sub_rev * C_rev + A_rev + eta_sub_rev * D_rev)
    if abs(denom_a_minus) < SMALL_EPSILON: denom_a_minus = np.complex128(SMALL_EPSILON + 0j)
    ra_minus_amp = (eta_inc_rev * B_rev + eta_inc_rev * eta_sub_rev * C_rev - A_rev - eta_sub_rev * D_rev) / denom_a_minus
    ta_minus_amp = (2.0 * eta_inc_rev) / denom_a_minus
    Ra_minus = abs(ra_minus_amp)**2; Ta_minus = 0.0
    if abs(real_eta_inc_rev) > SMALL_EPSILON: Ta_minus = (real_eta_sub_rev / real_eta_inc_rev) * abs(ta_minus_amp)**2

    # Combinaison incohérente (substrat épais)
    denom_global_fwd = 1.0 - Ra_minus * Rb_plus; denom_global_bwd = 1.0 - Ra_minus * Rb_minus
    if abs(denom_global_fwd) < SMALL_EPSILON: denom_global_fwd = SMALL_EPSILON
    if abs(denom_global_bwd) < SMALL_EPSILON: denom_global_bwd = SMALL_EPSILON

    R_global_calc = Ra_plus + (Ta_plus * Rb_plus * Ta_minus) / denom_global_fwd
    T_global_calc = (Ta_plus * Tb_plus) / denom_global_fwd
    R_prime_global_calc = Rb_plus + (Tb_plus * Ra_minus * Tb_minus) / denom_global_bwd # Réflexion côté substrat

    Rs_calc = max(0.0, min(1.0, R_global_calc.real)); Ts_calc = max(0.0, min(1.0, T_global_calc.real)); Rs_prime_calc = max(0.0, min(1.0, R_prime_global_calc.real))

    if not np.isfinite(Rs_calc): Rs_calc = np.nan
    if not np.isfinite(Ts_calc): Ts_calc = np.nan
    if not np.isfinite(Rs_prime_calc): Rs_prime_calc = np.nan

    return Rs_calc, Ts_calc, Rs_prime_calc


@numba.jit(nopython=True, cache=False)
def calculate_total_error_numba(l_array, nSub_array, target_value_array,
                                weights_array, target_type_flag,
                                current_thickness_nm,
                                n_calc_array, k_calc_array,
                                n_min_bound, n_max_bound, k_min_bound, k_max_bound):
    # ... (le reste de la fonction calculate_total_error_numba est inchangé)
    total_sq_error = 0.0
    points_calculated = 0
    for i in range(len(l_array)):
        l_val = l_array[i]
        nSub_val = nSub_array[i]
        # Ignorer le point si le substrat n'est pas défini à cette lambda
        if not np.isfinite(nSub_val): continue

        n_calc = n_calc_array[i]
        k_calc = k_calc_array[i]

        # Forcer les bornes physiques minimales
        n_calc = max(1.0, n_calc) # n ne peut pas être < 1
        k_calc = max(0.0, k_calc) # k ne peut pas être < 0

        # Pénalité si les valeurs n/k calculées sortent des bornes de l'optimiseur
        if not (n_min_bound <= n_calc <= n_max_bound): return HUGE_PENALTY
        if not (k_min_bound <= k_calc <= k_max_bound): return HUGE_PENALTY
        if not (np.isfinite(n_calc) and np.isfinite(k_calc)): return HUGE_PENALTY # Pénalité si NaN/inf

        nMono_complex_val = n_calc - 1j * k_calc

        # Calculer la transmission de l'empilement
        _, Ts_stack, _ = calculate_monolayer_lambda(l_val, nMono_complex_val, current_thickness_nm, nSub_val)
        if not np.isfinite(Ts_stack): continue # Ignorer si le calcul optique échoue

        calculated_value = np.nan
        if target_type_flag == 0: # T_norm
            # Calculer la transmission du substrat nu
            _, Ts_sub, _ = calculate_monolayer_lambda(l_val, 1.0 + 0j, 0.0, nSub_val) # n=1, k=0, d=0
            if not np.isfinite(Ts_sub): continue # Ignorer si le calcul optique échoue

            # Calculer T_norm
            if Ts_sub > SMALL_EPSILON: T_norm_calc = Ts_stack / Ts_sub
            else: T_norm_calc = 0.0 if abs(Ts_stack) < SMALL_EPSILON else HUGE_PENALTY

            if not np.isfinite(T_norm_calc): return HUGE_PENALTY # Pénalité si NaN/inf
            calculated_value = max(0.0, min(2.0, T_norm_calc)) # T_norm peut être > 1, mais limitons raisonnablement

        elif target_type_flag == 1: # T_sample
            calculated_value = max(0.0, min(1.0, Ts_stack))

        else: # Type inconnu
            return HUGE_PENALTY

        # Accumuler l'erreur quadratique si la cible est valide
        if np.isfinite(target_value_array[i]) and np.isfinite(calculated_value):
            error_i = (calculated_value - target_value_array[i])**2
            total_sq_error += error_i * weights_array[i]
            points_calculated += weights_array[i]
        elif not np.isfinite(target_value_array[i]):
             pass # Ignorer les points où la cible est NaN
        else: # Si calculated_value est NaN mais target ne l'est pas -> pénalité
            return HUGE_PENALTY

    # Retourner l'erreur quadratique moyenne
    if points_calculated <= SMALL_EPSILON: return HUGE_PENALTY # Éviter division par zéro
    return total_sq_error / points_calculated

# --- Fonction Objective pour l'Optimisation ---
def objective_func_spline_fixed_knots(p, num_knots_n, num_knots_k, l_array, nSub_array, target_value_array, weights_array, target_type_flag, fixed_n_knot_lambdas, fixed_k_knot_lambdas):
    # ... (le reste de la fonction objective_func_spline_fixed_knots est inchangé)
    expected_len = 1 + num_knots_n + num_knots_k
    if len(p) != expected_len: return HUGE_PENALTY # Vérification de base

    current_thickness_nm = p[0]
    idx_start = 1
    n_knot_values = p[idx_start : idx_start + num_knots_n]; idx_start += num_knots_n
    log_k_knot_values = p[idx_start : idx_start + num_knots_k]

    # Vérification rapide des bornes
    if current_thickness_nm < 0: return HUGE_PENALTY # Épaisseur négative
    if np.any(n_knot_values < N_KNOT_VALUE_BOUNDS[0] - SMALL_EPSILON) or \
       np.any(n_knot_values > N_KNOT_VALUE_BOUNDS[1] + SMALL_EPSILON) or \
       np.any(log_k_knot_values < LOG_K_KNOT_VALUE_BOUNDS[0] - SMALL_EPSILON) or \
       np.any(log_k_knot_values > LOG_K_KNOT_VALUE_BOUNDS[1] + SMALL_EPSILON): return HUGE_PENALTY

    try:
        # Interpolation Spline
        if num_knots_n < 2 or num_knots_k < 2 : return HUGE_PENALTY # Spline a besoin d'au moins 2 points
        n_spline = CubicSpline(fixed_n_knot_lambdas, n_knot_values, bc_type='natural', extrapolate=True)
        log_k_spline = CubicSpline(fixed_k_knot_lambdas, log_k_knot_values, bc_type='natural', extrapolate=True)

        # Évaluation des splines sur les longueurs d'onde cibles
        n_calc_array = n_spline(l_array)
        k_calc_array = np.exp(log_k_spline(l_array)) # k = exp(log_k)

        # Bornes physiques pour n/k utilisées dans la fonction numba
        k_min_req = math.exp(LOG_K_KNOT_VALUE_BOUNDS[0])
        k_max_req = math.exp(LOG_K_KNOT_VALUE_BOUNDS[1])

        # Calcul de l'erreur totale avec Numba
        mse = calculate_total_error_numba(l_array, nSub_array, target_value_array, weights_array, target_type_flag, current_thickness_nm, n_calc_array, k_calc_array, N_KNOT_VALUE_BOUNDS[0], N_KNOT_VALUE_BOUNDS[1], k_min_req, k_max_req)
        return mse

    except ValueError: # Erreur potentielle de CubicSpline (ex: valeurs non finies, non triées)
        return HUGE_PENALTY
    except Exception: # Autres erreurs inattendues
        return HUGE_PENALTY

# --- Fonctions de Logging et Affichage Streamlit ---
def add_log_message(message_type, message):
    if 'log_messages' not in st.session_state: st.session_state.log_messages = []
    st.session_state.log_messages.append((message_type, message))
    # Log aussi dans la console pour le débogage
    if message_type == "info": logger.info(message)
    elif message_type == "warning": logger.warning(message)
    elif message_type == "error": logger.error(message)

def display_log():
    if 'log_messages' in st.session_state and st.session_state.log_messages:
        with st.expander("Log Messages", expanded=False):
            log_container = st.container()
            # Afficher les messages les plus récents en premier
            for msg_type, msg in reversed(st.session_state.log_messages):
                if msg_type == "info": log_container.info(msg, icon="ℹ️")
                elif msg_type == "warning": log_container.warning(msg, icon="⚠️")
                elif msg_type == "error": log_container.error(msg, icon="🚨")
                else: log_container.text(msg)

def reset_log():
    st.session_state.log_messages = []

# --- Fonctions de Tracé Matplotlib ---
def plot_target_only(target_data_to_plot, target_filename_base):
    # ... (le reste de la fonction plot_target_only est inchangé)
    if target_data_to_plot is None or 'lambda' not in target_data_to_plot or len(target_data_to_plot['lambda']) == 0:
        add_log_message("warning", "No target data available to plot.")
        return None

    target_type = target_data_to_plot.get('target_type', 'T_norm')
    target_values = target_data_to_plot.get('target_value', None)
    target_l = target_data_to_plot['lambda']

    if target_values is None:
        add_log_message("warning", "Target values are missing in the data.")
        return None

    fig_cible, ax_cible = plt.subplots(1, 1, figsize=(8, 6));
    short_filename = target_filename_base if target_filename_base else "Target"
    if target_type == 'T_norm':
        plot_label = 'Target T Norm (%)'; y_label = 'Normalized Transmission (%)'; title_suffix = "T Norm (%)"; y_lim_top = 110
    else: # T_sample
        plot_label = 'Target T (%)'; y_label = 'Transmission (%)'; title_suffix = "T Sample (%)"; y_lim_top = 105

    ax_cible.set_title(f"Target Data ({short_filename}) - {title_suffix}")
    ax_cible.set_xlabel('λ (nm)');
    ax_cible.grid(True, which='both', linestyle=':', linewidth=0.5); ax_cible.minorticks_on();

    valid_mask = np.isfinite(target_values) & np.isfinite(target_l)
    if np.any(valid_mask):
        ax_cible.plot(target_l[valid_mask], target_values[valid_mask] * 100.0, '.', markersize=5, color='red', linestyle='none', label=plot_label)
        ax_cible.set_ylabel(y_label)
        # Ajustement dynamique de l'axe Y basé sur les données
        min_y_data = np.min(target_values[valid_mask] * 100.0)
        max_y_data = np.max(target_values[valid_mask] * 100.0)
        y_padding = max(5, (max_y_data - min_y_data) * 0.05) if max_y_data > min_y_data else 5
        ax_cible.set_ylim(bottom=max(-5, min_y_data - y_padding), top=min(y_lim_top + 10 , max_y_data + y_padding))
    else:
        add_log_message("warning", "No valid (finite) target data points to plot.")
        ax_cible.set_ylabel('No Valid Data')
        ax_cible.set_ylim(bottom=-5, top=y_lim_top) # Limites par défaut si pas de données

    # Limites de l'axe X basées sur les données du fichier ou les valeurs en mémoire
    lambda_min_plot = np.nanmin(target_l[valid_mask]) if 'lambda_min_file' not in st.session_state and np.any(valid_mask) else st.session_state.get('lambda_min_file', 0)
    lambda_max_plot = np.nanmax(target_l[valid_mask]) if 'lambda_max_file' not in st.session_state and np.any(valid_mask) else st.session_state.get('lambda_max_file', 1000)
    if lambda_min_plot < lambda_max_plot: ax_cible.set_xlim(lambda_min_plot, lambda_max_plot)

    ax_cible.legend(fontsize='small')
    plt.tight_layout()
    return fig_cible

def plot_nk_final(best_params_info, plot_lambda_array):
    # ... (le reste de la fonction plot_nk_final est inchangé)
    fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
    ax2 = ax1.twinx() # Axe Y partagé pour n et k

    optimal_thickness_nm = best_params_info['thickness_nm']
    n_values_opt_final = best_params_info['n_knot_values']
    log_k_values_opt_final = best_params_info['log_k_knot_values']
    fixed_n_lambdas = best_params_info['n_knot_lambdas']
    fixed_k_lambdas = best_params_info['k_knot_lambdas']
    num_n = best_params_info['num_knots_n']
    num_k = best_params_info['num_knots_k']
    lambda_min_eff = best_params_info['effective_lambda_min']
    lambda_max_eff = best_params_info['effective_lambda_max']

    try:
        if num_n < 2 or num_k < 2: raise ValueError("Need >= 2 knots for spline calculation.")
        n_spline = CubicSpline(fixed_n_lambdas, n_values_opt_final, bc_type='natural', extrapolate=True)
        log_k_spline = CubicSpline(fixed_k_lambdas, log_k_values_opt_final, bc_type='natural', extrapolate=True)
        n_plot = n_spline(plot_lambda_array); k_plot = np.exp(log_k_spline(plot_lambda_array))

        color1='tab:red'; color2='tab:blue' # Couleurs pour k et n

        # Axe K (gauche)
        ax1.set_xlabel('λ (nm)'); ax1.set_ylabel("k Coeff.", color=color1);
        ax1.plot(plot_lambda_array, k_plot, color=color1, linestyle='--', linewidth=1.5, label='k (Final)')
        ax1.plot(fixed_k_lambdas, np.exp(log_k_values_opt_final), 's', color=color1, markersize=6, fillstyle='none', label=f'k Knots ({num_k})')
        ax1.tick_params(axis='y', labelcolor=color1); ax1.grid(True,which='major',ls=':',lw=0.5,axis='y',color=color1)

        # Axe N (droite)
        ax2.set_ylabel('n Index', color=color2)
        ax2.plot(plot_lambda_array, n_plot, color=color2, linestyle='-', linewidth=1.5, label='n (Final)')
        ax2.plot(fixed_n_lambdas, n_values_opt_final, 'o', color=color2, markersize=6, fillstyle='none', label=f'n Knots ({num_n})')
        ax2.tick_params(axis='y', labelcolor=color2)

        # Grille X commune
        ax1.grid(True,which='both',ls=':',lw=0.5,axis='x');

        # Échelle Y pour K (log ou linéaire selon la plage)
        k_min_req = math.exp(LOG_K_KNOT_VALUE_BOUNDS[0]); k_max_req = math.exp(LOG_K_KNOT_VALUE_BOUNDS[1])
        min_k_plot_val = np.nanmin(k_plot) if k_plot is not None and np.any(np.isfinite(k_plot)) else k_min_req
        max_k_plot_val = np.nanmax(k_plot) if k_plot is not None and np.any(np.isfinite(k_plot)) else k_max_req
        min_k_disp = max(SMALL_EPSILON * 0.1, k_min_req * 0.5)
        max_k_disp = max(k_max_req * 2.0, max_k_plot_val * 1.2) if np.isfinite(max_k_plot_val) else k_max_req * 2.0

        use_log_k = (max_k_disp / min_k_disp) > 100 if min_k_disp > 0 and max_k_disp > 0 else False
        if use_log_k:
            ax1.set_yscale('log'); ax1.set_ylim(bottom=min_k_disp, top=max_k_disp); ax1.set_ylabel("k Coeff. [Log]", color=color1)
        else:
            ax1.set_yscale('linear'); k_range = k_max_req - k_min_req
            k_low_lim = max(0, k_min_req - 0.1 * k_range) if k_range > 0 else 0
            k_high_lim = k_max_req + 0.1 * k_range if k_range > 0 else k_max_req * 1.1
            ax1.set_ylim(bottom=k_low_lim, top=max(k_high_lim, max_k_plot_val * 1.1 if np.isfinite(max_k_plot_val) else k_high_lim)); ax1.set_ylabel("k Coeff.", color=color1)

        # Échelle Y pour N (linéaire)
        n_min_req = N_KNOT_VALUE_BOUNDS[0]; n_max_req = N_KNOT_VALUE_BOUNDS[1]
        n_range = n_max_req - n_min_req
        n_low_lim = n_min_req - 0.1 * n_range if n_range > 0 else n_min_req * 0.95
        n_high_lim = n_max_req + 0.1 * n_range if n_range > 0 else n_max_req * 1.05
        min_n_plot_val = np.nanmin(n_plot) if n_plot is not None and np.any(np.isfinite(n_plot)) else n_min_req
        max_n_plot_val = np.nanmax(n_plot) if n_plot is not None and np.any(np.isfinite(n_plot)) else n_max_req
        min_n_ylim = min(n_low_lim, min_n_plot_val * 0.98 if np.isfinite(min_n_plot_val) else n_low_lim)
        max_n_ylim = max(n_high_lim, max_n_plot_val * 1.02 if np.isfinite(max_n_plot_val) else n_high_lim)
        ax2.set_ylim(bottom=min_n_ylim, top=max_n_ylim)

        # Limites X
        if lambda_min_eff is not None and lambda_max_eff is not None and lambda_min_eff < lambda_max_eff :
            ax1.set_xlim(lambda_min_eff, lambda_max_eff)

        ax1.set_title('Final Optimized n/k Indices')
        # Légende combinée
        handles1, labels1 = ax1.get_legend_handles_labels(); handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc='best', fontsize='small')
        plt.tight_layout()
        return fig

    except Exception as e:
        add_log_message("error", f"Failed to generate final n/k plot: {e}")
        # traceback.print_exc() # Décommenter pour un débogage détaillé
        return None


def plot_spectra_vs_target(res, target=None, best_params_info=None, model_str_base="Spline Fit", effective_lambda_min=None, effective_lambda_max=None):
    # ... (le reste de la fonction plot_spectra_vs_target est inchangé)
    try:
        fig, ax = plt.subplots(1, 1, figsize=(9, 7)); ax_delta = ax.twinx() # Axe pour la différence

        # Construction du titre et labels
        num_knots_n = best_params_info.get('num_knots_n', '?'); num_knots_k = best_params_info.get('num_knots_k', '?'); knot_distrib = best_params_info.get('knot_distribution', '?')
        model_str = f"{model_str_base} ({num_knots_n}n/{num_knots_k}k Knots, {knot_distrib})"
        target_type = target.get('target_type', 'T_norm') if target else 'T_norm'

        if target_type == 'T_norm':
            comparison_label = 'Normalized T'; target_label_suffix = 'T Norm (%)'; calc_label_suffix = 'T Norm (%)'; y_label = 'Normalized Transmission (%)'; target_value_key = 'target_value'; calc_value_key = 'T_norm_calc'; y_lim_top = 110
        else: # T_sample
            comparison_label = 'Sample T'; target_label_suffix = 'T (%)'; calc_label_suffix = 'T (%)'; y_label = 'Transmission (%)'; target_value_key = 'target_value'; calc_value_key = 'T_stack_calc'; y_lim_top = 105

        # Titre détaillé
        title_base=f'{comparison_label} Comparison ({model_str})'; params_list = []
        if best_params_info and 'thickness_nm' in best_params_info: params_list.append(f"d={best_params_info['thickness_nm']:.2f}nm")
        if best_params_info and 'substrate_name' in best_params_info: params_list.append(f"Sub={best_params_info['substrate_name']}")
        if effective_lambda_min is not None and effective_lambda_max is not None: params_list.append(f"λ=[{effective_lambda_min:.0f}-{effective_lambda_max:.0f}]nm")
        if params_list: title_base += f' - {", ".join(params_list)}'
        title = title_base;
        # Ajout du MSE au titre s'il est disponible
        if 'MSE_Recalculated' in res and res['MSE_Recalculated'] is not None and np.isfinite(res['MSE_Recalculated']): title += f"\nFinal MSE (in Range): {res['MSE_Recalculated']:.3e}"
        elif 'MSE_Optimized' in res and res['MSE_Optimized'] is not None and np.isfinite(res['MSE_Optimized']): title += f"\nOptim. Objective MSE: {res['MSE_Optimized']:.3e}" # Fallback si MSE recalculé non dispo

        ax.set_title(title, fontsize=10);
        ax.set_xlabel('λ (nm)'); ax.set_ylabel(y_label); ax.grid(True, which='both', linestyle=':', linewidth=0.5); ax.minorticks_on(); ax.set_ylim(bottom=-5, top=y_lim_top)

        # Tracé du spectre calculé
        line_calc = None; calc_l = res.get('l'); calc_y = res.get(calc_value_key)
        if calc_l is not None and calc_y is not None:
            valid_calc_mask = np.isfinite(calc_y) & np.isfinite(calc_l)
            plot_mask = (calc_l >= effective_lambda_min) & (calc_l <= effective_lambda_max) & valid_calc_mask
            if np.any(plot_mask):
                line_calc, = ax.plot(calc_l[plot_mask], calc_y[plot_mask] * 100.0, label=f'Calc {calc_label_suffix}', linestyle='-', color='darkblue', linewidth=1.5);

        # Tracé de la cible (points)
        line_tgt = None; target_l_valid, target_y_valid = None, None
        if target is not None and 'lambda' in target and target_value_key in target and len(target['lambda']) > 0:
            target_l_valid_orig = target['lambda']; target_y_valid_fraction = target[target_value_key]
            valid_target_mask = np.isfinite(target_y_valid_fraction) & np.isfinite(target_l_valid_orig)
            plot_range_mask = (target_l_valid_orig >= effective_lambda_min) & (target_l_valid_orig <= effective_lambda_max)
            final_target_mask = valid_target_mask & plot_range_mask
            if np.any(final_target_mask):
                target_l_valid = target_l_valid_orig[final_target_mask]; target_y_valid = target_y_valid_fraction[final_target_mask] * 100.0
                line_tgt, = ax.plot(target_l_valid, target_y_valid, 'o', markersize=4, color='red', fillstyle='none', label=f'Target {target_label_suffix}');

        # Tracé de la différence (Calculé - Cible) sur l'axe droit
        line_delta = None; delta_t_perc = np.full_like(calc_l, np.nan) if calc_l is not None else np.array([])
        if calc_l is not None and calc_y is not None and target_l_valid is not None and target_y_valid is not None and len(target_l_valid) > 1: # Besoin d'assez de points cibles pour interpoler
            calc_y_perc = calc_y * 100.0
            # Interpoler la cible sur les longueurs d'onde du calcul pour pouvoir soustraire
            target_y_perc_interp = np.interp(calc_l, target_l_valid, target_y_valid, left=np.nan, right=np.nan)
            delta_t_perc = calc_y_perc - target_y_perc_interp

        valid_delta_mask = np.isfinite(delta_t_perc) & np.isfinite(calc_l)
        plot_mask_delta = (calc_l >= effective_lambda_min) & (calc_l <= effective_lambda_max) & valid_delta_mask
        if np.any(plot_mask_delta):
            line_delta, = ax_delta.plot(calc_l[plot_mask_delta], delta_t_perc[plot_mask_delta], label='ΔT (%) [Calc - Target]', linestyle=':', color='green', linewidth=1.2, zorder=-5);
            # Ajuster l'axe Y de la différence
            min_delta = np.min(delta_t_perc[plot_mask_delta]); max_delta = np.max(delta_t_perc[plot_mask_delta])
            padding = max(1.0, abs(max_delta - min_delta) * 0.1) if max_delta != min_delta else 1.0
            ax_delta.set_ylim(min_delta - padding, max_delta + padding); ax_delta.set_ylabel('ΔT (%) [Calc - Target]', color='green'); ax_delta.tick_params(axis='y', labelcolor='green'); ax_delta.grid(True, axis='y', linestyle='-.', linewidth=0.5, color='lightgreen', alpha=0.6)
        else:
             # Si pas de delta calculable, configurer l'axe quand même mais sans données/limites
            ax_delta.set_ylabel('ΔT (%) [Calc - Target]', color='green'); ax_delta.tick_params(axis='y', labelcolor='green'); ax_delta.set_yticks([])

        # Limites X
        if effective_lambda_min is not None and effective_lambda_max is not None and effective_lambda_min < effective_lambda_max:
            ax.set_xlim(effective_lambda_min, effective_lambda_max)
        else: # Fallback si la plage effective n'est pas définie
            min_l_plot_fallback = np.nanmin(res.get('l', [300])); max_l_plot_fallback = np.nanmax(res.get('l', [1000]));
            if np.isfinite(min_l_plot_fallback) and np.isfinite(max_l_plot_fallback) and min_l_plot_fallback < max_l_plot_fallback:
                ax.set_xlim(min_l_plot_fallback, max_l_plot_fallback)

        # Légende combinée pour les deux axes Y
        handles1, labels1 = ax.get_legend_handles_labels(); handles2, labels2 = ax_delta.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc='best', fontsize='small')

        # Affichage de la qualité du fit sur le graphe
        percent_good_fit = res.get('percent_good_fit', np.nan); quality_label = res.get('quality_label', 'N/A')
        if np.isfinite(percent_good_fit):
            quality_text = f"Fit Quality (in Range): {quality_label}\n(<0.25% abs delta): {percent_good_fit:.1f}%"
            # Positionner le texte en bas à droite du graphe
            ax.text(0.98, 0.02, quality_text, transform=ax.transAxes, fontsize=10, ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', alpha=0.8))

        plt.tight_layout()
        return fig
    except Exception as e_plot:
        add_log_message("error", f"Failed to generate final spectra plot: {e_plot}")
        # traceback.print_exc() # Décommenter pour un débogage détaillé
        return None

def plot_substrate_indices():
    # ... (le reste de la fonction plot_substrate_indices est inchangé)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    lambda_nm = np.linspace(200, 2000, 500) # Plage de longueurs d'onde pour le tracé

    for idx, name in enumerate(SUBSTRATE_LIST):
        min_wl_sub = get_substrate_min_lambda(name)
        try:
            n_values = np.array([get_n_substrate(idx, l) for l in lambda_nm])
            # Tracer la partie valide en ligne pleine
            valid_mask = lambda_nm >= min_wl_sub
            line, = ax.plot(lambda_nm[valid_mask], n_values[valid_mask], label=f"{name} (≥{min_wl_sub:.0f} nm)", linewidth=1.5)
            # Tracer la partie invalide (extrapolation ou hors plage) en pointillé
            invalid_mask = lambda_nm < min_wl_sub
            if np.any(invalid_mask):
                # Recalculer pour la partie invalide pour éviter les erreurs si get_n_substrate retourne NaN
                n_invalid = np.array([get_n_substrate(idx, l) for l in lambda_nm[invalid_mask]])
                # Utiliser la même couleur que la ligne principale
                ax.plot(lambda_nm[invalid_mask], n_invalid, linestyle=':', color=line.get_color(), alpha=0.5)

        except Exception as e:
            add_log_message("warning", f"Could not calculate index for {name}. Error: {e}")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Refractive Index (n)")
    ax.set_title("Substrate Refractive Indices (Sellmeier - Dashed below nominal range)")
    ax.grid(True, which='major', linestyle='-', linewidth='0.5', color='gray')
    ax.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgray')
    ax.minorticks_on()
    ax.legend()
    ax.set_xlim(200, 2000) # Limites fixes pour la comparaison
    plt.tight_layout()
    return fig

def draw_schema_matplotlib(target_type, substrate_name):
    # ... (le reste de la fonction draw_schema_matplotlib est inchangé)
    fig, ax = plt.subplots(figsize=(5.5, 1.5)) # Ajuster la taille au besoin
    fig.patch.set_alpha(0) # Fond transparent
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis('off') # Pas d'axes visibles

    # Couleurs et polices
    color_Mono = "#D6EAF8"; color_Sub = "#EAECEE"; outline_color = "#5D6D7E"
    layer_font_size = 7; medium_font_size = 7
    layer_font = {'size': layer_font_size, 'color': outline_color}
    medium_font = {'size': medium_font_size, 'weight': 'bold', 'color': outline_color}

    # Dimensions et positions
    x_left = 10; stack_width = 25; mono_h = 8; sub_h = 8; y_sub_base = 5

    # Dessin de la couche et du substrat (à gauche ou unique si T_sample)
    ax.add_patch(plt.Rectangle((x_left, y_sub_base + sub_h), stack_width, mono_h, facecolor=color_Mono, edgecolor=outline_color, linewidth=0.5))
    ax.text(x_left + stack_width/2, y_sub_base + sub_h + mono_h/2, "Monolayer", ha='center', va='center', fontdict=layer_font)
    ax.add_patch(plt.Rectangle((x_left, y_sub_base), stack_width, sub_h, facecolor=color_Sub, edgecolor=outline_color, linewidth=0.5))
    ax.text(x_left + stack_width/2, y_sub_base + sub_h/2, f"Sub ({substrate_name})", ha='center', va='center', fontdict=layer_font)

    # Milieux environnants (Air)
    ax.text(x_left + stack_width/2, y_sub_base + sub_h + mono_h + 3, "Air (n≈1)", ha='center', va='top', fontdict=medium_font)
    ax.text(x_left + stack_width/2, y_sub_base - 3, "Air", ha='center', va='bottom', fontdict=medium_font)

    # Flèche et label pour la transmission (T_sample ou T_norm)
    arrow_x = x_left + stack_width/2; y_arrow_start = 28; y_arrow_end = 2
    ax.arrow(arrow_x, y_arrow_start, 0, y_arrow_end - y_arrow_start, head_width=3, head_length=2, fc='darkred', ec='darkred', length_includes_head=True, width=0.5)
    label_text = "T_sample" if target_type == 'T' else "T_norm"
    ax.text(arrow_x + 4, y_arrow_end + 5, label_text, ha='left', va='center', color='darkred', style='italic', size=8)

    # Section supplémentaire pour T_norm (substrat nu à droite)
    if target_type == 'T_norm':
        x_right = 100 - 10 - stack_width # Position à droite

        # Dessin du substrat nu
        ax.add_patch(plt.Rectangle((x_right, y_sub_base), stack_width, sub_h, facecolor=color_Sub, edgecolor=outline_color, linewidth=0.5))
        ax.text(x_right + stack_width/2, y_sub_base + sub_h/2, f"Sub ({substrate_name})", ha='center', va='center', fontdict=layer_font)

        # Milieux Air
        ax.text(x_right + stack_width/2, y_sub_base + sub_h + 3, "Air (n≈1)", ha='center', va='top', fontdict=medium_font)
        ax.text(x_right + stack_width/2, y_sub_base - 3, "Air", ha='center', va='bottom', fontdict=medium_font)

        # Flèche et label pour T_sub
        arrow_x_right = x_right + stack_width / 2
        ax.arrow(arrow_x_right, y_arrow_start, 0, y_arrow_end - y_arrow_start, head_width=3, head_length=2, fc='darkred', ec='darkred', length_includes_head=True, width=0.5)
        ax.text(arrow_x_right + 4, y_arrow_end + 5, "T_sub", ha='left', va='center', color='darkred', style='italic', size=8)

        # Texte explicatif pour T_norm au centre
        x_center = 50
        text_tnorm = f"{label_text} = T_sample / T_sub"
        # text_tnorm = "considering this\nautozero/baseline\nconfiguration" # Texte original, peut-être moins clair
        ax.text(x_center, 15, text_tnorm, ha='center', va='center', style='italic', size=7, color=outline_color)

    return fig

# --- Fonction Export Excel ---
def create_excel_file(results_data):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_summary = results_data['summary']
        df_results = results_data['data']
        # Utiliser un nom de feuille court et informatif
        sheet_name_short = results_data.get('sheet_name', 'Results')[:30] # Limiter la longueur du nom de feuille

        # Écrire le résumé en premier
        df_summary.to_excel(writer, index=False, sheet_name=sheet_name_short, startrow=0, startcol=0)

        # Écrire les données détaillées après le résumé, avec un espace
        df_results.to_excel(writer, index=False, sheet_name=sheet_name_short, startrow=len(df_summary)+2, startcol=0)

        # (Optionnel) Ajuster la largeur des colonnes pour une meilleure lisibilité
        # worksheet = writer.sheets[sheet_name_short]
        # for col in df_results:
        #     max_len = max(df_results[col].astype(str).map(len).max(), len(col)) + 2
        #     worksheet.column_dimensions[get_column_letter(df_results.columns.get_loc(col) + 1)].width = max_len

    output.seek(0) # Rembobiner le buffer pour la lecture
    return output.getvalue()

# --- Initialisation de l'état de session Streamlit ---
# Utiliser st.session_state pour conserver les valeurs entre les re-runs
if 'target_data' not in st.session_state: st.session_state.target_data = None
if 'target_filename_base' not in st.session_state: st.session_state.target_filename_base = None
if 'lambda_min_file' not in st.session_state: st.session_state.lambda_min_file = None
if 'lambda_max_file' not in st.session_state: st.session_state.lambda_max_file = None
if 'log_messages' not in st.session_state: st.session_state.log_messages = []
if 'optim_results' not in st.session_state: st.session_state.optim_results = None
if 'excel_bytes' not in st.session_state: st.session_state.excel_bytes = None
if 'excel_filename' not in st.session_state: st.session_state.excel_filename = "results.xlsx"
if 'config_lambda_min' not in st.session_state: st.session_state.config_lambda_min = "---"
if 'config_lambda_max' not in st.session_state: st.session_state.config_lambda_max = "---"
if 'thickness_min' not in st.session_state: st.session_state.thickness_min = 300.0
if 'thickness_max' not in st.session_state: st.session_state.thickness_max = 600.0
if 'substrate_choice' not in st.session_state: st.session_state.substrate_choice = SUBSTRATE_LIST[0]
if 'target_type' not in st.session_state: st.session_state.target_type = "T_norm"
if 'last_loaded_source' not in st.session_state: st.session_state.last_loaded_source = None # Pour suivre si défaut ou upload

# Paramètres avancés par défaut
default_advanced_params = {
    'num_knots_n': 6, 'num_knots_k': 6, 'use_inv_lambda_sq_distrib': False,
    'pop_size': 20, 'maxiter': 1500, 'tol': 0.001, 'atol': 0.0,
    'mutation_min': 0.5, 'mutation_max': 1.2, 'recombination': 0.8,
    'strategy': 'best1bin', 'polish': True, 'updating': 'deferred', 'workers': -1
}
if 'advanced_optim_params' not in st.session_state:
    st.session_state.advanced_optim_params = default_advanced_params.copy()

# --- Interface Utilisateur Streamlit ---
st.set_page_config(page_title="Monolayer Optimizer", layout="wide") # Nom de l'onglet

# --- Titre et Explication Intégrée ---
st.title("Monolayer Optical Properties Optimizer")

with st.expander("📄 Description de l'application", expanded=False): # 'expanded=False' pour qu'il soit fermé par défaut
    st.markdown(r"""
    Cette application Streamlit détermine les propriétés optiques (indice de réfraction $n(\lambda)$, coefficient d'extinction $k(\lambda)$) et l'épaisseur physique ($d$) d'une couche mince unique (monocouche) déposée sur un substrat connu.

    La détermination est réalisée en ajustant des spectres de transmission optique calculés (soit la transmission normalisée $T_{norm}$ soit la transmission directe de l'échantillon $T_{sample}$) à des données cibles expérimentales fournies par l'utilisateur via un fichier CSV (ou un fichier d'exemple par défaut).
    L'application utilise des splines cubiques pour modéliser la dispersion de $n(\lambda)$ et $k(\lambda)$, et emploie l'algorithme d'évolution différentielle pour trouver les paramètres optimaux (valeurs des nœuds des splines et épaisseur $d$) qui minimisent l'erreur quadratique moyenne entre le spectre calculé et la cible expérimentale dans une gamme de longueurs d'onde définie par l'utilisateur, tout en respectant les limitations physiques du substrat choisi.

    L'outil produit les dispersions déterminées de $n(\lambda)$ et $k(\lambda)$, l'épaisseur optimale $d$, des graphiques de comparaison, une évaluation de la qualité de l'ajustement, et permet d'exporter des résultats détaillés dans un fichier Excel.

    ---
    *Développé par Fabien Lemarchand.*
    *Pour toute question, retour d'information ou collaboration potentielle, veuillez contacter :* [fabien.lemarchand@gmail.com](mailto:fabien.lemarchand@gmail.com)
    """)

# --- Sidebar pour la Configuration ---
with st.sidebar:
    st.header("Configuration")
    prev_substrate = st.session_state.substrate_choice
    st.session_state.substrate_choice = st.selectbox(
        "Substrate Material:", SUBSTRATE_LIST,
        index=SUBSTRATE_LIST.index(st.session_state.substrate_choice) if st.session_state.substrate_choice in SUBSTRATE_LIST else 0
    )
    # Mise à jour auto de lambda min si le substrat change ET qu'un fichier est chargé
    if st.session_state.substrate_choice != prev_substrate and st.session_state.target_data is not None:
        substrate_min_wl = get_substrate_min_lambda(st.session_state.substrate_choice)
        if st.session_state.lambda_min_file is not None:
            new_config_min = max(st.session_state.lambda_min_file, substrate_min_wl)
            try:
                current_max = float(st.session_state.config_lambda_max) if st.session_state.config_lambda_max != "---" else np.inf
                if new_config_min < current_max:
                    st.session_state.config_lambda_min = f"{new_config_min:.1f}"
                    add_log_message("info", f"Substrate changed to {st.session_state.substrate_choice}. Default Min Lambda updated to {st.session_state.config_lambda_min} nm.")
                else:
                    add_log_message("warning", f"Substrate changed. New suggested min lambda ({new_config_min:.1f} nm) >= current max lambda ({st.session_state.config_lambda_max}). Manual adjustment needed.")
            except ValueError: # Au cas où config_lambda_max n'est pas un float valide
                 st.session_state.config_lambda_min = f"{new_config_min:.1f}"
                 add_log_message("info", f"Substrate changed to {st.session_state.substrate_choice}. Default Min Lambda updated to {st.session_state.config_lambda_min} nm.")
            st.rerun() # Met à jour l'affichage de la sidebar

    st.subheader("Optimization Lambda Range")
    col_lam1, col_lam2 = st.columns(2)
    with col_lam1:
        st.session_state.config_lambda_min = st.text_input("Min λ (nm):", value=st.session_state.config_lambda_min, help=f"Must be >= substrate limit ({get_substrate_min_lambda(st.session_state.substrate_choice):.1f} nm) and >= file min λ.")
    with col_lam2:
        st.session_state.config_lambda_max = st.text_input("Max λ (nm):", value=st.session_state.config_lambda_max, help="Must be <= file max λ.")

    # Validation simple des entrées lambda
    try:
        lmin = float(st.session_state.config_lambda_min) if st.session_state.config_lambda_min != "---" else -1
        lmax = float(st.session_state.config_lambda_max) if st.session_state.config_lambda_max != "---" else -1
        sub_min = get_substrate_min_lambda(st.session_state.substrate_choice)
        if lmin != -1 and lmin < sub_min: st.warning(f"Min λ ({lmin:.1f}) is below substrate limit ({sub_min:.1f})!", icon="⚠️")
        if lmin != -1 and lmax != -1 and lmin >= lmax: st.warning("Min λ must be less than Max λ!", icon="⚠️")
        if st.session_state.lambda_min_file is not None and lmin != -1 and lmin < st.session_state.lambda_min_file: st.warning(f"Min λ ({lmin:.1f}) is below file minimum ({st.session_state.lambda_min_file:.1f})!", icon="⚠️")
        if st.session_state.lambda_max_file is not None and lmax != -1 and lmax > st.session_state.lambda_max_file: st.warning(f"Max λ ({lmax:.1f}) is above file maximum ({st.session_state.lambda_max_file:.1f})!", icon="⚠️")
    except ValueError:
         if st.session_state.config_lambda_min != "---" or st.session_state.config_lambda_max != "---": st.warning("Invalid numeric format for Lambda Min/Max.", icon="⚠️")

    # Bouton pour afficher le graphe des indices substrat
    if st.button("Plot Substrate Indices"):
        fig_sub = plot_substrate_indices()
        if fig_sub: st.session_state.fig_substrate_plot = fig_sub
        else: st.error("Could not generate substrate plot.")

    # Paramètres avancés dans un expander
    with st.expander("Advanced Optimization Settings"):
        adv_params = st.session_state.advanced_optim_params
        adv_params['num_knots_n'] = st.number_input("n Spline Knots", min_value=2, value=adv_params['num_knots_n'], step=1)
        adv_params['num_knots_k'] = st.number_input("k Spline Knots", min_value=2, value=adv_params['num_knots_k'], step=1)
        adv_params['use_inv_lambda_sq_distrib'] = st.checkbox("Knot Distribution 1/λ² (else 1/λ)", value=adv_params['use_inv_lambda_sq_distrib'])
        st.markdown("---")
        st.subheader("Differential Evolution")
        adv_params['pop_size'] = st.number_input("Population Size (popsize)", min_value=5, value=adv_params['pop_size'], step=5)
        adv_params['maxiter'] = st.number_input("Max Iterations (maxiter)", min_value=1, value=adv_params['maxiter'], step=100)
        col_tol1a, col_tol2a = st.columns(2)
        with col_tol1a: adv_params['tol'] = st.number_input("Relative Tol (tol)", min_value=0.0, value=adv_params['tol'], format="%.4f", step=0.001)
        with col_tol2a: adv_params['atol'] = st.number_input("Absolute Tol (atol)", min_value=0.0, value=adv_params['atol'], format="%.4f", step=0.001)
        col_mut1a, col_mut2a = st.columns(2)
        with col_mut1a: adv_params['mutation_min'] = st.number_input("Mutation Min", min_value=0.0, max_value=2.0, value=adv_params['mutation_min'], step=0.1)
        with col_mut2a: adv_params['mutation_max'] = st.number_input("Mutation Max", min_value=adv_params['mutation_min'], max_value=2.0, value=adv_params['mutation_max'], step=0.1)
        adv_params['recombination'] = st.slider("Recombination", min_value=0.0, max_value=1.0, value=adv_params['recombination'], step=0.1)
        adv_params['polish'] = st.checkbox("Polish final solution", value=adv_params['polish'])
        strategy_options = ['best1bin', 'best1exp', 'rand1exp', 'randtobest1exp', 'currenttobest1exp', 'best2exp', 'rand2exp', 'randtobest1bin', 'currenttobest1bin', 'best2bin', 'rand2bin', 'rand1bin']
        adv_params['strategy'] = st.selectbox("DE Strategy", options=strategy_options, index=strategy_options.index(adv_params['strategy']))
        updating_options = ['immediate', 'deferred']
        adv_params['updating'] = st.selectbox("Updating Mode", options=updating_options, index=updating_options.index(adv_params['updating']))
        adv_params['workers'] = st.number_input("Parallel Workers (-1 = Auto)", min_value=-1, value=adv_params['workers'], step=1, help="Utilise plusieurs coeurs CPU si > 1 ou -1. Peut ne pas fonctionner sur toutes les plateformes Streamlit Cloud.")

# --- Colonnes Principales ---
col1, col2 = st.columns([0.6, 0.4]) # 60% pour les contrôles/plots, 40% pour le schéma/substrat

with col1:
    st.subheader("Target Data")
    st.session_state.target_type = st.radio(
        "Select Target Type:",
        options=["T_norm", "T"],
        format_func=lambda x: "T Norm (%) = T_sample / T_sub" if x == "T_norm" else "T Sample (%)",
        index=["T_norm", "T"].index(st.session_state.target_type),
        horizontal=True,
        key="target_type_radio"
    )

    # Widget pour charger un fichier utilisateur
    uploaded_file = st.file_uploader(
        "Upload Target File (.csv) or use default:",
        type=["csv"],
        accept_multiple_files=False,
        key="file_uploader"
    )

    # Définir le chemin du fichier par défaut
    default_file_path = 'example.csv'
    data_source_to_load = None # Contiendra soit uploaded_file soit default_file_path
    source_name = None         # Nom du fichier (uploadé ou par défaut)
    is_default_file = False

    if uploaded_file is not None:
        # Priorité au fichier uploadé par l'utilisateur
        if uploaded_file.name != st.session_state.get('last_loaded_source', None):
            data_source_to_load = uploaded_file
            source_name = uploaded_file.name
            st.session_state.last_loaded_source = source_name # Mémoriser le nom du fichier chargé
            add_log_message("info", f"User uploaded file: {source_name}")
            reset_log() # Effacer les logs précédents lors d'un nouveau chargement
            st.session_state.optim_results = None # Réinitialiser les résultats précédents
            st.session_state.excel_bytes = None
    elif st.session_state.target_data is None: # Si aucun fichier uploadé ET aucune donnée déjà chargée
        # Essayer de charger le fichier par défaut UNIQUEMENT si aucune donnée n'est présente
        add_log_message("info", f"No file uploaded. Attempting to load default: {default_file_path}")
        if os.path.exists(default_file_path):
            if default_file_path != st.session_state.get('last_loaded_source', None):
                data_source_to_load = default_file_path
                source_name = default_file_path
                st.session_state.last_loaded_source = source_name
                is_default_file = True
                reset_log() # Effacer les logs précédents
                st.session_state.optim_results = None # Réinitialiser les résultats précédents
                st.session_state.excel_bytes = None
            # else: Fichier défaut déjà chargé et mémorisé, ne rien faire
        else:
            # Fichier défaut non trouvé et pas d'upload -> Afficher message qu'après
            if st.session_state.last_loaded_source != "DEFAULT_NOT_FOUND": # Évite répétition message
                 add_log_message("error", f"Default file '{default_file_path}' not found. Please upload a file.")
                 st.warning(f"Default file '{default_file_path}' not found. Please upload a file.", icon="⚠️")
                 st.session_state.last_loaded_source = "DEFAULT_NOT_FOUND"

    # --- Logique de parsing (commune pour upload et défaut) ---
    if data_source_to_load is not None:
        st.session_state.target_data = None # Réinitialiser avant de parser
        try:
            file_extension = os.path.splitext(source_name)[1].lower()
            if file_extension != ".csv": raise ValueError(f"Unsupported file extension: '{file_extension}'. Please select a .csv file.")

            add_log_message("info", f"Parsing {source_name}...")
            df = None
            error_messages = []

            # Tentatives de lecture CSV (adapté de ton code original)
            parse_attempts = [
                {'delimiter': ',', 'decimal': '.', 'encoding': 'utf-8', 'msg': "Parse attempt: Delimiter=',' Decimal='.' Encoding='utf-8'"},
                {'delimiter': ',', 'decimal': '.', 'encoding': 'latin-1', 'msg': "Parse attempt: Delimiter=',' Decimal='.' Encoding='latin-1'"},
                {'delimiter': ',', 'decimal': ',', 'encoding': 'utf-8', 'msg': "Parse attempt: Delimiter=',' Decimal=',' Encoding='utf-8'"},
                {'delimiter': ',', 'decimal': ',', 'encoding': 'latin-1', 'msg': "Parse attempt: Delimiter=',' Decimal=',' Encoding='latin-1'"},
                # Ajouter d'autres combinaisons si nécessaire (ex: point-virgule)
                # {'delimiter': ';', 'decimal': ',', 'encoding': 'utf-8', 'msg': "Parse attempt: Delimiter=';' Decimal=',' Encoding='utf-8'"},
                # {'delimiter': ';', 'decimal': '.', 'encoding': 'utf-8', 'msg': "Parse attempt: Delimiter=';' Decimal='.' Encoding='utf-8'"},
            ]

            for attempt in parse_attempts:
                try:
                    # Utiliser data_source_to_load qui est soit UploadedFile soit un path
                    add_log_message("info", attempt['msg'])
                    df_attempt = pd.read_csv(
                        data_source_to_load,
                        delimiter=attempt['delimiter'],
                        decimal=attempt['decimal'],
                        skiprows=1,         # Ignorer l'en-tête
                        usecols=[0, 1],     # Utiliser seulement les 2 premières colonnes
                        names=['lambda', 'target_value'], # Nommer les colonnes
                        encoding=attempt['encoding'],
                        skipinitialspace=True # Ignorer les espaces après le délimiteur
                    )
                    # Vérification simple si le parsing a fonctionné (au moins une ligne et les colonnes attendues)
                    if df_attempt is not None and not df_attempt.empty and 'lambda' in df_attempt.columns and 'target_value' in df_attempt.columns:
                        # Essayer de convertir en numérique pour détecter les erreurs tôt
                        pd.to_numeric(df_attempt['lambda'], errors='raise')
                        pd.to_numeric(df_attempt['target_value'], errors='raise')
                        df = df_attempt # Succès!
                        add_log_message("info", f"Successfully parsed with: {attempt}")
                        break # Sortir de la boucle des tentatives
                    else:
                         # Rembobiner si c'est un objet fichier uploadé
                        if hasattr(data_source_to_load, 'seek'): data_source_to_load.seek(0)

                except Exception as e_parse:
                    error_messages.append(f"Attempt ({attempt}) failed: {e_parse}")
                    add_log_message("warning", f"Parse attempt ({attempt['msg']}) failed: {e_parse}")
                    # Rembobiner si c'est un objet fichier uploadé
                    if hasattr(data_source_to_load, 'seek'): data_source_to_load.seek(0)

            if df is None:
                raise ValueError(f"Could not read or parse the CSV file '{source_name}' after multiple attempts. Please check format (delimiter, decimal, encoding). Errors: {'; '.join(error_messages)}")

            if df.empty: raise ValueError("CSV file loaded but appears empty or has incorrect structure after skipping header.")
            if 'lambda' not in df.columns or 'target_value' not in df.columns: raise ValueError("Could not find expected columns ('lambda', 'target_value'). Check header/columns.")

            # Conversion en numérique et gestion des erreurs
            df['lambda'] = pd.to_numeric(df['lambda'], errors='coerce'); df['target_value'] = pd.to_numeric(df['target_value'], errors='coerce')
            if df['lambda'].isnull().any(): add_log_message("warning", f"Non-numeric values found in lambda column, replaced with NaN.")
            if df['target_value'].isnull().any(): add_log_message("warning", f"Non-numeric values found in target value column, replaced with NaN.")

            # Préparation des données numpy et tri
            data = df.to_numpy(); data = data[data[:, 0].argsort()]; lam = data[:, 0]; target_value_raw = data[:, 1]

            # Validations supplémentaires
            if len(lam) < 3: raise ValueError("Not enough data rows (< 3).")
            valid_lam_mask = np.isfinite(lam)
            if not np.any(valid_lam_mask): raise ValueError("No valid finite wavelength values found.")
            min_lam_val = np.min(lam[valid_lam_mask])
            if min_lam_val <= 0: raise ValueError("Wavelengths (λ) must be > 0.")

            valid_target_mask = np.isfinite(target_value_raw)
            if not np.any(valid_target_mask): raise ValueError("No valid finite target values found.")

            # Conversion % -> fraction si nécessaire
            valid_targets = target_value_raw[valid_target_mask]
            if len(valid_targets) > 0 and np.nanmax(valid_targets) > 5: # Seuil arbitraire pour détecter les %
                target_value_final = target_value_raw / 100.0
                add_log_message("info", f"Target Data values > 5 detected, interpreting as % and dividing by 100.")
            else:
                target_value_final = target_value_raw
                add_log_message("info", f"Target Data values <= 5, interpreting as fraction (0-1).")

            # Stocker les données dans session_state
            st.session_state.target_data = {'lambda': lam, 'target_value': target_value_final, 'target_type': st.session_state.target_type}
            valid_finite_lam = lam[valid_lam_mask]
            st.session_state.lambda_min_file = np.min(valid_finite_lam); st.session_state.lambda_max_file = np.max(valid_finite_lam)
            if st.session_state.lambda_min_file >= st.session_state.lambda_max_file - SMALL_EPSILON: raise ValueError("Invalid wavelength range in file after processing (Min >= Max).")

            st.session_state.target_filename_base = source_name # Mémoriser le nom du fichier utilisé

            # Mise à jour de la plage lambda par défaut dans la sidebar
            current_substrate = st.session_state.substrate_choice; substrate_min_wl = get_substrate_min_lambda(current_substrate); initial_config_min = max(st.session_state.lambda_min_file, substrate_min_wl)
            if initial_config_min >= st.session_state.lambda_max_file:
                add_log_message("warning", f"The minimum allowed wavelength ({initial_config_min:.1f} nm) for substrate {current_substrate} is >= max wavelength in file ({st.session_state.lambda_max_file:.1f} nm). Cannot set a valid default range.")
                st.session_state.config_lambda_min = "---"; st.session_state.config_lambda_max = "---"
            else:
                st.session_state.config_lambda_min = f"{initial_config_min:.1f}"; st.session_state.config_lambda_max = f"{st.session_state.lambda_max_file:.1f}";
                add_log_message("info", f"Set/Update default optimization range based on file and substrate ({current_substrate}): [{initial_config_min:.1f}, {st.session_state.lambda_max_file:.1f}] nm")

            # Afficher le graphe cible
            fig_target = plot_target_only(st.session_state.target_data, st.session_state.target_filename_base)
            if fig_target: st.session_state.fig_target_plot = fig_target

            # Infos de log finales sur le chargement
            num_total_pts = len(lam); num_valid_lambda = np.sum(valid_lam_mask); num_valid_target = np.sum(valid_target_mask)
            log_source = "Default file" if is_default_file else "Uploaded file"
            add_log_message("info", f"{log_source} '{st.session_state.target_filename_base}' loaded ({num_total_pts} rows).")
            add_log_message("info", f"  Valid lambda points: {num_valid_lambda}/{num_total_pts}.")
            add_log_message("info", f"  Valid target points: {num_valid_target}/{num_total_pts}.")
            if num_valid_lambda > 0: add_log_message("info", f"  λ range in file: [{st.session_state.lambda_min_file:.1f}, {st.session_state.lambda_max_file:.1f}] nm.")
            add_log_message("info", f"  Target type assumed: {st.session_state.target_type}.")
            if num_valid_lambda < num_total_pts or num_valid_target < num_total_pts: add_log_message("warning", f"Some rows contained invalid non-numeric values (NaN).")

            st.rerun() # Rafraîchir l'interface avec les nouvelles données et la plage lambda

        except (ValueError, ImportError, Exception) as e:
            st.error(f"Error reading or processing file '{source_name}':\n{e}", icon="🚨"); add_log_message("error", f"ERROR loading file '{source_name}': {e}")
            # traceback.print_exc() # Pour débogage
            st.session_state.target_data = None; st.session_state.lambda_min_file = None; st.session_state.lambda_max_file = None;
            st.session_state.target_filename_base = "Error Loading"; st.session_state.config_lambda_min = "---"; st.session_state.config_lambda_max = "---"; st.session_state.fig_target_plot = None
            st.session_state.last_loaded_source = f"ERROR_{source_name}" # Éviter re-tentative de chargement erroné
            st.rerun() # Rafraîchir pour afficher l'erreur


    # Afficher le statut du fichier chargé (ou erreur) et le graphe cible si disponible
    if st.session_state.target_filename_base:
        if st.session_state.target_filename_base == "Error Loading":
            # L'erreur est déjà affichée par le bloc try/except
            pass
        elif st.session_state.target_data is not None:
             file_source_msg = "(Default)" if st.session_state.last_loaded_source == default_file_path else "(Uploaded)"
             st.success(f"Loaded: {st.session_state.target_filename_base} {file_source_msg} (Type: {st.session_state.target_type})", icon="✅")
             if 'fig_target_plot' in st.session_state and st.session_state.fig_target_plot is not None:
                 st.pyplot(st.session_state.fig_target_plot)
             else: # Au cas où le graphe n'a pas pu être généré mais les données sont là
                 st.warning("Target data loaded, but plot could not be generated.", icon="⚠️")
    elif st.session_state.last_loaded_source != "DEFAULT_NOT_FOUND": # Si aucun fichier chargé et pas d'erreur fichier défaut
        st.info("Upload a CSV target file or ensure 'example.csv' is present for default loading.")

    st.subheader("Thickness Configuration")
    col_thick1a, col_thick2a = st.columns(2)
    with col_thick1a: st.session_state.thickness_min = st.number_input("Min Thick (nm):", min_value=0.0, value=st.session_state.thickness_min, step=10.0, format="%.1f")
    with col_thick2a: st.session_state.thickness_max = st.number_input("Max Thick (nm):", min_value=max(0.1, st.session_state.thickness_min + 0.1), value=st.session_state.thickness_max, step=10.0, format="%.1f") # Assure Max > Min

    st.divider()
    run_button = st.button("▶ Run Optimization", type="primary", use_container_width=True, disabled=(st.session_state.target_data is None))

with col2:
    st.subheader("Comparative Schema")
    # Le schéma se met à jour automatiquement grâce au st.radio et st.selectbox qui déclenchent un rerun
    fig_schema = draw_schema_matplotlib(st.session_state.target_type, st.session_state.substrate_choice)
    st.pyplot(fig_schema)

    # Afficher le graphe substrat s'il a été généré
    if 'fig_substrate_plot' in st.session_state and st.session_state.fig_substrate_plot is not None:
        st.subheader("Substrate Indices")
        st.pyplot(st.session_state.fig_substrate_plot)
        if st.button("Clear Substrate Plot"):
            st.session_state.fig_substrate_plot = None;
            st.rerun()

# --- Exécution de l'Optimisation ---
if run_button:
    reset_log(); st.session_state.optim_results = None; st.session_state.excel_bytes = None
    add_log_message("info", "="*20 + " Starting Optimization " + "="*20)
    valid_params = True
    try:
        # Récupérer et valider les paramètres
        thickness_min = float(st.session_state.thickness_min); thickness_max = float(st.session_state.thickness_max)
        if thickness_min > thickness_max or thickness_min < 0: raise ValueError("Invalid Monolayer Thickness bounds (Min >= 0, Min <= Max).")

        lambda_min_str = st.session_state.config_lambda_min; lambda_max_str = st.session_state.config_lambda_max
        if not lambda_min_str or not lambda_max_str or lambda_min_str == "---" or lambda_max_str == "---": raise ValueError("Lambda Min/Max not set. Load a file and check sidebar configuration.")
        effective_lambda_min = float(lambda_min_str); effective_lambda_max = float(lambda_max_str)
        if effective_lambda_min <= 0 or effective_lambda_max <= 0: raise ValueError("Lambda Min/Max must be positive.")
        if effective_lambda_min >= effective_lambda_max: raise ValueError("Lambda Min must be less than Max Lambda.")

        if st.session_state.lambda_min_file is None or st.session_state.lambda_max_file is None: raise ValueError("File lambda range not available. Load file again.")
        if effective_lambda_min < st.session_state.lambda_min_file - SMALL_EPSILON or effective_lambda_max > st.session_state.lambda_max_file + SMALL_EPSILON: raise ValueError(f"Optimization lambda range [{effective_lambda_min:.1f}, {effective_lambda_max:.1f}] nm must be within the loaded file's range [{st.session_state.lambda_min_file:.1f}, {st.session_state.lambda_max_file:.1f}] nm.")

        selected_substrate = st.session_state.substrate_choice; substrate_min_limit = get_substrate_min_lambda(selected_substrate)
        if effective_lambda_min < substrate_min_limit:
             add_log_message("warning", f"Specified Min Lambda ({effective_lambda_min:.1f} nm) is below minimum for {selected_substrate} ({substrate_min_limit:.1f} nm). Adjusting Min Lambda to substrate limit.")
             effective_lambda_min = substrate_min_limit; st.session_state.config_lambda_min = f"{effective_lambda_min:.1f}" # Met à jour la session state aussi
             if effective_lambda_min >= effective_lambda_max: raise ValueError(f"Adjusted Min Lambda ({effective_lambda_min:.1f} nm) is now >= Max Lambda ({effective_lambda_max:.1f} nm). Check configuration.")
             st.rerun() # Important pour afficher la valeur ajustée avant de continuer l'optimisation

        add_log_message("info", f"Using optimization range: [{effective_lambda_min:.1f}, {effective_lambda_max:.1f}] nm")
        current_target_type = st.session_state.target_type; target_type_flag = 0 if current_target_type == 'T_norm' else 1
        substrate_id = SUBSTRATE_LIST.index(selected_substrate)
        adv_params = st.session_state.advanced_optim_params; num_knots_n = adv_params['num_knots_n']; num_knots_k = adv_params['num_knots_k']; use_inv_lambda_sq_distrib = adv_params['use_inv_lambda_sq_distrib']

    except (ValueError, TypeError, KeyError) as e_param: st.error(f"Parameter Error: Invalid parameter value:\n{e_param}\nCheck Thickness and Sidebar settings.", icon="🚨"); add_log_message("error", f"Parameter Error: {e_param}"); valid_params = False
    except Exception as e_unexpected: st.error(f"Setup Error: An unexpected error occurred during setup:\n{e_unexpected}", icon="🚨"); add_log_message("error", f"Unexpected Setup Error: {e_unexpected}"); valid_params = False

    if valid_params:
        add_log_message("info", f"Starting optimization: Target={current_target_type}, Substrate={selected_substrate}")
        add_log_message("info", f"Thickness Range: [{thickness_min:.1f}, {thickness_max:.1f}] nm"); add_log_message("info", f"Advanced parameters: {adv_params}")

        # Filtrer les données cibles pour l'optimisation
        current_target_lambda = st.session_state.target_data['lambda']; current_target_value = st.session_state.target_data['target_value']
        lambda_range_mask = (current_target_lambda >= effective_lambda_min) & (current_target_lambda <= effective_lambda_max)
        valid_target_mask_finite = np.isfinite(current_target_value) & np.isfinite(current_target_lambda)
        nSub_target_array_full = np.array([get_n_substrate(substrate_id, l) for l in current_target_lambda])
        valid_substrate_mask = np.isfinite(nSub_target_array_full)

        mask_used_in_optimization = valid_target_mask_finite & lambda_range_mask & valid_substrate_mask

        if not np.any(mask_used_in_optimization):
            st.error(f"No valid target data points found within the specified lambda range [{effective_lambda_min:.1f}, {effective_lambda_max:.1f}] nm with a valid substrate index.", icon="🚨"); add_log_message("error", "No valid points for optimization in the specified range.")
        else:
            target_lambda_opt = current_target_lambda[mask_used_in_optimization]; target_value_opt = current_target_value[mask_used_in_optimization]; nSub_target_array_opt = nSub_target_array_full[mask_used_in_optimization]
            weights_array = np.ones_like(target_lambda_opt); # Poids uniformes pour l'instant
            add_log_message("info", f"Using {len(target_lambda_opt)} target points for optimization.")

            # Calcul des positions des nœuds fixes pour les splines
            fixed_n_knot_lambdas = np.array([], dtype=float); fixed_k_knot_lambdas = np.array([], dtype=float); knot_lam_min = effective_lambda_min; knot_lam_max = effective_lambda_max
            try:
                if use_inv_lambda_sq_distrib:
                    # Distribution en 1/lambda^2
                    inv_lambda_sq_min = 1.0 / (knot_lam_max**2); inv_lambda_sq_max = 1.0 / (knot_lam_min**2)
                    if num_knots_n > 0: fixed_n_knot_lambdas = 1.0 / np.sqrt(np.linspace(inv_lambda_sq_min, inv_lambda_sq_max, num_knots_n) + SMALL_EPSILON)
                    if num_knots_k > 0: fixed_k_knot_lambdas = 1.0 / np.sqrt(np.linspace(inv_lambda_sq_min, inv_lambda_sq_max, num_knots_k) + SMALL_EPSILON)
                else:
                    # Distribution en 1/lambda (linéaire en fréquence/énergie)
                    inv_lambda_min = 1.0 / knot_lam_max; inv_lambda_max = 1.0 / knot_lam_min
                    if num_knots_n > 0: fixed_n_knot_lambdas = 1.0 / (np.linspace(inv_lambda_min, inv_lambda_max, num_knots_n) + SMALL_EPSILON)
                    if num_knots_k > 0: fixed_k_knot_lambdas = 1.0 / (np.linspace(inv_lambda_min, inv_lambda_max, num_knots_k) + SMALL_EPSILON)

                # Assurer que les nœuds sont dans la plage et triés (clip + sort)
                fixed_n_knot_lambdas = np.clip(np.sort(fixed_n_knot_lambdas), knot_lam_min + SMALL_EPSILON, knot_lam_max - SMALL_EPSILON)
                fixed_k_knot_lambdas = np.clip(np.sort(fixed_k_knot_lambdas), knot_lam_min + SMALL_EPSILON, knot_lam_max - SMALL_EPSILON)

                # Vérifier les doublons (peut arriver si plage très étroite)
                if len(np.unique(fixed_n_knot_lambdas)) < num_knots_n or len(np.unique(fixed_k_knot_lambdas)) < num_knots_k:
                    add_log_message("warning", "Duplicate knot wavelengths generated, likely due to narrow range or low knot count. Adjusting slightly using linspace as fallback.")
                    fixed_n_knot_lambdas = np.linspace(knot_lam_min + SMALL_EPSILON, knot_lam_max - SMALL_EPSILON, num_knots_n); fixed_k_knot_lambdas = np.linspace(knot_lam_min + SMALL_EPSILON, knot_lam_max - SMALL_EPSILON, num_knots_k)

            except Exception as e_knot: st.error(f"Knot Error: Error calculating knot positions for range [{knot_lam_min:.1f}, {knot_lam_max:.1f}] nm:\n{e_knot}", icon="🚨"); add_log_message("error", f"Knot calculation error: {e_knot}"); valid_params = False

            if valid_params:
                # Définir les bornes pour l'optimiseur
                parameter_bounds = [(thickness_min, thickness_max)] + [N_KNOT_VALUE_BOUNDS] * num_knots_n + [LOG_K_KNOT_VALUE_BOUNDS] * num_knots_k

                # Arguments fixes pour la fonction objective
                fixed_args = (num_knots_n, num_knots_k, target_lambda_opt, nSub_target_array_opt, target_value_opt, weights_array, target_type_flag, fixed_n_knot_lambdas, fixed_k_knot_lambdas)

                # Callback pour suivre la progression
                optim_iteration_count = [0]
                optim_callback_best_mse = [np.inf]
                status_text = st.empty() # Placeholder pour afficher la progression

                def optimization_callback_simple_log(xk, convergence):
                    optim_iteration_count[0] += 1
                    # Évaluer l'objectif actuel (peut être coûteux, faire avec parcimonie)
                    # Afficher toutes les N itérations ou si meilleur MSE trouvé
                    display_freq = 50
                    is_best = False
                    if optim_iteration_count[0] % display_freq == 0 or optim_iteration_count[0] == 1:
                        try:
                             current_fun = objective_func_spline_fixed_knots(xk, *fixed_args)
                             if not np.isfinite(current_fun): current_fun = np.inf
                             if current_fun < optim_callback_best_mse[0]:
                                 optim_callback_best_mse[0] = current_fun; is_best = True

                             mse_val = optim_callback_best_mse[0]
                             status_text.info(f"Iteration: {optim_iteration_count[0]} | Best MSE: {mse_val:.4e}", icon="⏳")
                             # Log plus détaillé dans le log expander
                             if is_best: add_log_message("info", f"Iter: {optim_iteration_count[0]}, New Best MSE: {mse_val:.4e}")
                             else: add_log_message("info", f"Iter: {optim_iteration_count[0]}, Current MSE: {current_fun:.4e}, Best MSE: {mse_val:.4e}")
                        except Exception as e_cb:
                            add_log_message("warning", f"Error in callback at iter {optim_iteration_count[0]}: {e_cb}")


                # Arguments pour differential_evolution
                de_args = {
                    'func': objective_func_spline_fixed_knots,
                    'bounds': parameter_bounds,
                    'args': fixed_args,
                    'strategy': adv_params['strategy'],
                    'maxiter': adv_params['maxiter'],
                    'popsize': adv_params['pop_size'],
                    'tol': adv_params['tol'],
                    'atol': adv_params['atol'],
                    'mutation': (adv_params['mutation_min'], adv_params['mutation_max']),
                    'recombination': adv_params['recombination'],
                    'polish': adv_params['polish'],
                    'updating': adv_params['updating'],
                    'workers': adv_params['workers'],
                    'disp': False, # Ne pas afficher la sortie de scipy dans la console
                    'callback': optimization_callback_simple_log
                 }

                try:
                    with st.spinner("Optimization running... Please wait."):
                        start_time_opt = time.time(); optim_result = scipy.optimize.differential_evolution(**de_args); end_time_opt = time.time()
                    status_text.success(f"Optimization finished in {end_time_opt - start_time_opt:.2f} s.", icon="✅"); add_log_message("info", f"Optimization finished in {end_time_opt - start_time_opt:.2f} s.")

                    if not optim_result.success: add_log_message("warning", f"Main optimization did not converge successfully: {optim_result.message}"); st.warning(f"Optimization warning: {optim_result.message}", icon="⚠️")

                    add_log_message("info", "-" * 50); add_log_message("info", f"Best result from optimization:");
                    p_optimal = optim_result.x; final_objective_value = optim_result.fun; final_mse_display = final_objective_value if np.isfinite(final_objective_value) and final_objective_value < HUGE_PENALTY else np.nan
                    add_log_message("info", f"  Optimal MSE (Objective Func): {final_mse_display:.4e}")

                    optimal_thickness_nm = p_optimal[0]; add_log_message("info", f"  Optimal Monolayer Thickness: {optimal_thickness_nm:.3f} nm");

                    # Extraire les valeurs optimales des nœuds
                    idx_start = 1; n_values_opt_final = p_optimal[idx_start : idx_start + num_knots_n]; idx_start += num_knots_n; log_k_values_opt_final = p_optimal[idx_start : idx_start + num_knots_k]

                    # Recalculer n, k, T_stack, T_norm sur TOUTE la plage lambda du fichier original
                    n_spline_final = CubicSpline(fixed_n_knot_lambdas, n_values_opt_final, bc_type='natural', extrapolate=True); log_k_spline_final = CubicSpline(fixed_k_knot_lambdas, log_k_values_opt_final, bc_type='natural', extrapolate=True)
                    nSub_target_array_recalc = np.array([get_n_substrate(substrate_id, l) for l in current_target_lambda]) # nSub sur toute la plage

                    n_final_array_recalc = np.full_like(current_target_lambda, np.nan); k_final_array_recalc = np.full_like(current_target_lambda, np.nan); T_stack_final_calc = np.full_like(current_target_lambda, np.nan); T_norm_final_calc = np.full_like(current_target_lambda, np.nan)

                    # Masque pour évaluer les splines uniquement là où lambda est valide
                    valid_lambda_mask_for_calc = (current_target_lambda >= substrate_min_limit) & np.isfinite(current_target_lambda) & (current_target_lambda > 0)
                    if np.any(valid_lambda_mask_for_calc):
                         n_final_array_recalc[valid_lambda_mask_for_calc] = n_spline_final(current_target_lambda[valid_lambda_mask_for_calc]);
                         k_final_array_recalc[valid_lambda_mask_for_calc] = np.exp(log_k_spline_final(current_target_lambda[valid_lambda_mask_for_calc]))

                         # Clipser aux bornes physiques/optimiseur après extrapolation potentielle
                         n_final_array_recalc = np.clip(n_final_array_recalc, 1.0, N_KNOT_VALUE_BOUNDS[1]);
                         k_final_array_recalc = np.clip(k_final_array_recalc, 0.0, math.exp(LOG_K_KNOT_VALUE_BOUNDS[1]))
                         n_final_array_recalc[~np.isfinite(n_final_array_recalc)] = np.nan; k_final_array_recalc[~np.isfinite(k_final_array_recalc)] = np.nan

                    # Calculer Tstack et Tnorm là où n, k, et nSub sont valides
                    valid_nk_final_mask = np.isfinite(n_final_array_recalc) & np.isfinite(k_final_array_recalc); valid_nsub_mask_recalc = np.isfinite(nSub_target_array_recalc);
                    valid_indices_for_T_calc = np.where(valid_nk_final_mask & valid_nsub_mask_recalc)[0]

                    for i in valid_indices_for_T_calc:
                        l_val = current_target_lambda[i]; nMono_val = n_final_array_recalc[i] - 1j * k_final_array_recalc[i]; nSub_val = nSub_target_array_recalc[i]
                        try:
                            _, Ts_stack_calc, _ = calculate_monolayer_lambda(l_val, nMono_val, optimal_thickness_nm, nSub_val); _, Ts_sub_calc, _ = calculate_monolayer_lambda(l_val, 1.0 + 0j, 0.0, nSub_val) # Substrat nu
                            if np.isfinite(Ts_stack_calc): T_stack_final_calc[i] = np.clip(Ts_stack_calc, 0.0, 1.0)
                            T_norm_calc = np.nan
                            if np.isfinite(Ts_sub_calc):
                                if Ts_sub_calc > SMALL_EPSILON: T_norm_calc = Ts_stack_calc / Ts_sub_calc
                                elif abs(Ts_stack_calc) < SMALL_EPSILON : T_norm_calc = 0.0 # Si les deux sont proches de 0
                            if np.isfinite(T_norm_calc): T_norm_final_calc[i] = np.clip(T_norm_calc, 0.0, 2.0) # Tnorm peut dépasser 1
                        except Exception: pass # Ignorer les erreurs de calcul isolées

                    # Recalculer le MSE final et la qualité DANS LA PLAGE D'OPTIMISATION
                    calc_value_for_mse = T_norm_final_calc if current_target_type == 'T_norm' else T_stack_final_calc;
                    valid_calc_mask_recalc = np.isfinite(calc_value_for_mse)
                    combined_valid_mask_for_mse = mask_used_in_optimization & valid_calc_mask_recalc; # Utiliser le masque d'optimisation original
                    recalc_mse_final = np.nan; percent_good_fit = np.nan; quality_label = "N/A"; mse_pts_count = np.sum(combined_valid_mask_for_mse)

                    if mse_pts_count > 0 :
                        recalc_mse_final = np.mean((calc_value_for_mse[combined_valid_mask_for_mse] - current_target_value[combined_valid_mask_for_mse])**2);
                        # Calcul de la qualité du fit (pourcentage de points avec delta < seuil)
                        abs_delta = np.abs(calc_value_for_mse[combined_valid_mask_for_mse] - current_target_value[combined_valid_mask_for_mse]); delta_threshold = 0.0025 # Seuil de 0.25%
                        points_below_threshold = np.sum(abs_delta < delta_threshold);
                        percent_good_fit = (points_below_threshold / mse_pts_count) * 100.0

                        # Étiquette de qualité
                        if percent_good_fit >= 90: quality_label = "Excellent";
                        elif percent_good_fit >= 70: quality_label = "Good";
                        elif percent_good_fit >= 50: quality_label = "Fair";
                        else: quality_label = "Poor"

                        add_log_message("info", f"  Final MSE ({current_target_type}, {mse_pts_count} pts in range): {recalc_mse_final:.4e}"); add_log_message("info", "-"*20 + " Fit Quality " + "-"*20)
                        add_log_message("info", f"  Range [{effective_lambda_min:.1f}-{effective_lambda_max:.1f}] nm, {mse_pts_count} valid pts"); add_log_message("info", f"  Points with |delta| < {delta_threshold*100:.2f}% : {percent_good_fit:.1f}% ({points_below_threshold}/{mse_pts_count})"); add_log_message("info", f"  -> Rating: {quality_label}")
                    else:
                        add_log_message("warning", f"Cannot recalculate Final MSE or Fit Quality for range [{effective_lambda_min:.1f}-{effective_lambda_max:.1f}] nm. No valid points found after recalculation.")
                    add_log_message("info", "-" * 50)

                    # Stocker tous les résultats pour l'affichage et l'export
                    plot_lambda_array_final = np.linspace(effective_lambda_min, effective_lambda_max, 500) # Pour tracer n/k
                    st.session_state.optim_results = {
                        'final_spectra': {
                            'l': current_target_lambda, # Lambda sur toute la plage du fichier
                            'T_stack_calc': T_stack_final_calc, # Calcul sur toute la plage
                            'T_norm_calc': T_norm_final_calc,   # Calcul sur toute la plage
                            'MSE_Optimized': final_mse_display,
                            'MSE_Recalculated': recalc_mse_final, # MSE dans la plage d'opti
                            'percent_good_fit': percent_good_fit, # Qualité dans la plage d'opti
                            'quality_label': quality_label        # Label dans la plage d'opti
                        },
                        'target_filtered_for_plot': { # Cible uniquement dans la plage d'opti pour le graphe
                            'lambda': current_target_lambda[mask_used_in_optimization],
                            'target_value': current_target_value[mask_used_in_optimization],
                            'target_type': current_target_type
                        },
                        'best_params': {
                            'thickness_nm': optimal_thickness_nm,
                            'num_knots_n': num_knots_n, 'num_knots_k': num_knots_k,
                            'n_knot_values': n_values_opt_final, 'log_k_knot_values': log_k_values_opt_final,
                            'n_knot_lambdas': fixed_n_knot_lambdas, 'k_knot_lambdas': fixed_k_knot_lambdas,
                            'knot_distribution': "1/λ²" if use_inv_lambda_sq_distrib else "1/λ",
                            'substrate_name': selected_substrate,
                            'effective_lambda_min': effective_lambda_min, # Plage effectivement utilisée
                            'effective_lambda_max': effective_lambda_max
                        },
                        'plot_lambda_array': plot_lambda_array_final, # Lambda pour tracer n/k
                        'model_str_base': "Spline Fit",
                        # Données pour l'export Excel (sur toute la plage du fichier)
                        'excel_export_data': {
                            'lambda (nm)': current_target_lambda,
                            f'n (Spline Fit ({num_knots_n}n{num_knots_k}k))': n_final_array_recalc,
                            f'k (Spline Fit ({num_knots_n}n{num_knots_k}k))': k_final_array_recalc,
                            'Thickness (nm)': np.full_like(current_target_lambda, optimal_thickness_nm),
                            f'n Substrate ({selected_substrate})': nSub_target_array_recalc,
                            # Afficher la cible uniquement dans la plage d'opti dans Excel pour clarté
                            f'Target {current_target_type} (%) (Used)': np.where(mask_used_in_optimization, current_target_value * 100.0, np.nan),
                            f'Target {current_target_type} (%) (Full File)': current_target_value * 100.0, # Nouvelle colonne
                            'Calc T (%)': T_stack_final_calc * 100.0,
                            'Calc T Norm (%)': T_norm_final_calc * 100.0,
                            'Delta T (%)': (T_stack_final_calc - current_target_value)*100.0 if current_target_type == 'T' else np.nan, # Ajout Delta
                            'Delta T Norm (%)': (T_norm_final_calc - current_target_value)*100.0 if current_target_type == 'T_norm' else np.nan, # Ajout Delta
                        },
                        'excel_summary_params': [
                            ('Model', f"Spline Fit ({'1/λ²' if use_inv_lambda_sq_distrib else '1/λ'})"),
                            ('Target Used', current_target_type),
                            ('Target File', st.session_state.target_filename_base),
                            ('Substrate', selected_substrate),
                            ('Optimization Lambda Range (nm)', f"[{effective_lambda_min:.1f}, {effective_lambda_max:.1f}]"),
                            ('n Knots', num_knots_n), ('k Knots', num_knots_k),
                            ('Thickness (nm)', f"{optimal_thickness_nm:.3f}"),
                            ('Optim. MSE (Objective)', f"{final_mse_display:.4e}" if np.isfinite(final_mse_display) else "N/A"),
                            ('Final MSE (Recalc. in Range)', f"{recalc_mse_final:.4e}" if np.isfinite(recalc_mse_final) else "N/A"),
                            ('Fit Rating (in Range)', quality_label),
                            ('Pts in Range (|delta|<0.25%)', f"{percent_good_fit:.1f}%" if np.isfinite(percent_good_fit) else "N/A"),
                            # Détails des noeuds (optionnel, peut être long)
                            # ('n Knot λ (nm)', ", ".join([f"{v:.1f}" for v in fixed_n_knot_lambdas])),
                            # ('n Knot Values (opti)', ", ".join([f"{v:.4f}" for v in n_values_opt_final])),
                            # ('k Knot λ (nm)', ", ".join([f"{v:.1f}" for v in fixed_k_knot_lambdas])),
                            # ('k Knot Values (log, opti)', ", ".join([f"{v:.4f}" for v in log_k_values_opt_final]))
                        ]
                    }

                    # Générer le fichier Excel en mémoire
                    try:
                        excel_data_prep = {
                            'summary': pd.DataFrame(st.session_state.optim_results['excel_summary_params'], columns=['Parameter', 'Value']),
                            'data': pd.DataFrame(st.session_state.optim_results['excel_export_data']),
                            'sheet_name': f"Results_{selected_substrate[:10]}" # Nom de feuille court
                        }
                        # Formatage des nombres dans le DataFrame de données
                        df_res = excel_data_prep['data']; float_format = "%.5f" # Plus de décimales pour n/k/T
                        for col in df_res.select_dtypes(include=['float']).columns:
                             if col == 'lambda (nm)': df_res[col] = df_res[col].apply(lambda x: "%.2f" % x if pd.notna(x) else '')
                             elif col == 'Thickness (nm)': df_res[col] = df_res[col].apply(lambda x: "%.3f" % x if pd.notna(x) else '')
                             elif 'Target' in col or 'Calc T' in col or 'Delta T' in col: df_res[col] = df_res[col].apply(lambda x: "%.2f" % x if pd.notna(x) else '')
                             else: df_res[col] = df_res[col].apply(lambda x: float_format % x if pd.notna(x) else '') # n, k, nSub

                        excel_data_prep['data'] = df_res
                        st.session_state.excel_bytes = create_excel_file(excel_data_prep)

                        # Nom de fichier Excel plus descriptif
                        knot_distrib_str_short = "1L2" if use_inv_lambda_sq_distrib else "1L"
                        safe_target_filename = "".join(c if c.isalnum() else "_" for c in st.session_state.target_filename_base.split('.')[0])[:20]
                        st.session_state.excel_filename = f"results_{safe_target_filename}_{num_knots_n}n{num_knots_k}k_{knot_distrib_str_short}_{current_target_type}_{selected_substrate.replace(' ','_')}_L{effective_lambda_min:.0f}-{effective_lambda_max:.0f}.xlsx"
                        add_log_message("info", f"Excel data generated for download as '{st.session_state.excel_filename}'.")
                    except Exception as e_excel: add_log_message("error", f"Failed to generate Excel data in memory: {e_excel}"); st.error(f"Failed to generate Excel data: {e_excel}", icon="🚨"); st.session_state.excel_bytes = None

                except Exception as e_optim: st.error(f"Optimization Error: An error occurred during optimization:\n{e_optim}", icon="🚨"); add_log_message("error", f"ERROR during optimization: {e_optim}"); traceback.print_exc() # Imprimer la trace pour le débogage

# --- Affichage des Résultats (si disponibles) ---
if st.session_state.optim_results:
    results = st.session_state.optim_results
    st.divider(); st.header("Optimization Results")
    col_res1a, col_res2a = st.columns(2)
    with col_res1a: st.metric("Optimal Thickness", f"{results['best_params']['thickness_nm']:.3f} nm")
    with col_res2a:
        mse_disp = results['final_spectra']['MSE_Recalculated'];
        st.metric("Final MSE (in range)", f"{mse_disp:.4e}" if np.isfinite(mse_disp) else "N/A")

    # Afficher la qualité du fit
    quality_label = results['final_spectra']['quality_label']
    percent_good_fit = results['final_spectra']['percent_good_fit']
    if np.isfinite(percent_good_fit):
        st.metric("Fit Quality Rating (in range)", f"{quality_label}", help=f"Based on {percent_good_fit:.1f}% points within optim. range having |Calc - Target| < 0.25%")
    else:
        st.metric("Fit Quality Rating (in range)", "N/A")

    st.subheader("Result Plots")
    # Passer la cible filtrée pour le tracé de comparaison
    fig_compare = plot_spectra_vs_target(
        res=results['final_spectra'],
        target=st.session_state.target_data, # Passer toutes les données cibles originales
        # target=results['target_filtered_for_plot'], # <- Ancienne version: seulement les points utilisés
        best_params_info=results['best_params'],
        model_str_base=results['model_str_base'],
        effective_lambda_min=results['best_params']['effective_lambda_min'],
        effective_lambda_max=results['best_params']['effective_lambda_max']
    )
    if fig_compare: st.pyplot(fig_compare)

    # Tracer n/k
    fig_nk = plot_nk_final(results['best_params'], results['plot_lambda_array'])
    if fig_nk: st.pyplot(fig_nk)

    st.subheader("Result Data")
    if st.session_state.excel_bytes:
        st.download_button(
            label="💾 Download Results (.xlsx)",
            data=st.session_state.excel_bytes,
            file_name=st.session_state.excel_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    with st.expander("Show Result Data Table (Full Range)"):
        if 'excel_export_data' in results:
            # Afficher le DataFrame préparé pour Excel (déjà formaté)
            st.dataframe(pd.DataFrame(results['excel_export_data']).set_index('lambda (nm)'))
        else: st.warning("Result data not available for display.")

# Afficher les logs à la fin
display_log()

# --- Section Aide / Manuel Utilisateur ---
help_text_en = """
User Manual - Optical Monolayer Optimizer (Streamlit Version)

Goal:
This program determines the optical properties (refractive index n, extinction coefficient k) and thickness (d) of a single thin film (monolayer) deposited on a known substrate. It adjusts these parameters so that the calculated transmission best matches experimental target data.

Main Steps:

1.  **Configure Settings (Sidebar):**
    * **Substrate Material:** Choose the substrate from the dropdown. This affects the refractive index used in calculations and sets a minimum valid wavelength.
    * **Optimization Lambda Range:** Define the Min/Max wavelength (nm) for the optimization. This range must be within the range of your loaded file AND above the substrate's minimum valid wavelength. Values are automatically suggested after loading a file but can be adjusted. Check sidebar warnings.
    * **Advanced Optimization Settings (Optional):** Expand this section to fine-tune spline parameters (knots) and Differential Evolution algorithm settings (for advanced users). Defaults are usually reasonable.

2.  **Select Target Type (Main Area):**
    * Choose whether your target data represents "Normalized Transmission" (T Norm (%)) or "Sample Transmission" (T Sample (%)).
    * *T Norm (%)* = (Transmission of the sample with layer) / (Transmission of the bare substrate) * 100
    * *T Sample (%)* = Transmission of the sample with layer * 100
    * The schema diagram updates to reflect the selected measurement configuration.

3.  **Load Target File (Main Area):**
    * Click "Browse files" and select a **.csv** file containing your experimental data.
    * **Alternatively:** If no file is selected, the application will attempt to load a default file named `example.csv` if it exists in the same directory as the script.
    * **CSV Expected Format:**
        * First Line: Header (ignored). Data starts on the second line.
        * Columns: Column 1: Wavelength λ (nm), Column 2: Target value (T Norm or T Sample).
        * **Separators/Encoding:** The app attempts to automatically detect common formats (comma/semicolon delimiters, period/comma decimals, UTF-8/Latin-1 encoding). Ensure your data is numeric.
    * Target values can be percentage (e.g., 95.5) or fraction (e.g., 0.955). The program attempts to detect this automatically (values > 5 are assumed to be %).
    * A plot of the loaded data (target vs. λ) appears after successful loading.
    * The filename, its valid wavelength range, and suggested optimization range in the sidebar are updated. Check the Log Messages expander at the bottom for details on loading.

4.  **Configure Thickness Range (Main Area):**
    * Define the **Min/Max Thick (nm)** range for the monolayer thickness search. Ensure Min < Max.

5.  **Run Optimization (Main Area):**
    * Click the "▶ Run Optimization" button (enabled only after data is loaded).
    * A spinner indicates the process is running. Progress messages (Iteration/MSE) may appear below the button.
    * The Log Messages section at the bottom provides detailed information about the optimization steps.
    * This step performs the search for the best n(λ), k(λ), and thickness 'd' matching the target data within the specified range.

6.  **Analyze Results (Main Area - appears after optimization):**
    * **Metrics:** Optimal Thickness, Final Mean Squared Error (MSE) within the optimization range, and a Fit Quality Rating (Excellent/Good/Fair/Poor based on points close to the target) are displayed.
    * **Result Plots:**
        * *Comparison Plot:* Compares target data (red dots) with the calculated spectrum (blue line) over the **full** wavelength range of the input file. The difference ΔT is shown on the right axis (green dotted line) *only within the optimization range*. The Fit Quality text box refers to the quality *within the optimization range*.
        * *Final n/k Plot:* Shows the optimal n(λ) (blue) and k(λ) (red, potentially log scale) curves and the calculated spline knot positions (circles/squares) *within the optimization range*.
    * **Result Data:**
        * A "Download Results (.xlsx)" button appears to save a detailed Excel file containing parameters, n/k values, calculated spectra, and target data over the full range.
        * An expander ("Show Result Data Table") allows viewing the result data table directly in the app.

Tips:
- If loading fails, check the CSV format (delimiters, decimals, numeric data) and review the Log Messages.
- Choose a realistic thickness range. A too-wide range might slow down optimization unnecessarily.
- The optimization Lambda range (set in the sidebar) is crucial: it defines the data used for fitting AND must respect the substrate's physical limits. Check sidebar warnings.
- A low MSE value and a good 'Fit Quality' indicate a successful fit *mathematically*.
- **Crucially:** Visually inspect the Comparison Plot (does the blue line match the red dots well?) AND the n/k Plot (are the resulting n(λ) and k(λ) curves physically plausible for your material?). Sometimes a good mathematical fit yields unphysical optical constants.
- If using parallel workers (`workers > 1` or `-1`), ensure your environment supports multiprocessing (may not work on all free Streamlit Cloud tiers).
"""

with st.expander("Help / Instructions", expanded=False):
    st.markdown(help_text_en)

# Footer/Info dans la sidebar
st.sidebar.markdown("---")
st.sidebar.info("Monolayer Optimizer vX.Y - Streamlit App adapted from original code by F. Lemarchand.")
