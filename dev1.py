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
import csv
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

USER_LOG_FILE = 'user_log.csv'
SMALL_EPSILON = 1e-9
PI = np.pi
HUGE_PENALTY = 1e30
N_KNOT_VALUE_BOUNDS = (1.2, 4.0)
LOG_K_KNOT_VALUE_BOUNDS = (math.log(1e-6), math.log(5.0))
SUBSTRATE_LIST = ["SiO2", "N-BK7", "D263T eco", "Sapphire-fresnel", "B270i"]
SUBSTRATE_MIN_LAMBDA = {
    "SiO2": 230.0, "N-BK7": 400.0, "D263T eco": 360.0,
    "Sapphire-fresnel": 230.0, "B270i": 400.0,
}
DEFAULT_EXAMPLE_FILE = 'example.csv'

def get_substrate_min_lambda(substrate_name):
    return SUBSTRATE_MIN_LAMBDA.get(substrate_name, 200.0)

@numba.jit(nopython=True, cache=True)
def sellmeier_calc(l_um_sq, B1, C1, B2, C2, B3, C3):
    n_sq_minus_1 = 0.0
    if abs(C1) > SMALL_EPSILON and abs(l_um_sq - C1) > SMALL_EPSILON:
        n_sq_minus_1 += (B1 * l_um_sq / (l_um_sq - C1))
    if abs(C2) > SMALL_EPSILON and abs(l_um_sq - C2) > SMALL_EPSILON:
        n_sq_minus_1 += (B2 * l_um_sq / (l_um_sq - C2))
    if abs(C3) > SMALL_EPSILON and abs(l_um_sq - C3) > SMALL_EPSILON:
        n_sq_minus_1 += (B3 * l_um_sq / (l_um_sq - C3))
    n_sq = n_sq_minus_1 + 1.0
    if n_sq < 0: n_sq = 0
    return np.sqrt(n_sq)

@numba.jit(nopython=True, cache=True)
def get_n_substrate(substrate_id, wavelength_nm):
    l_um = wavelength_nm / 1000.0
    l_um_sq = l_um * l_um
    if substrate_id == 0:
        min_wl = 230.0
        if wavelength_nm < min_wl: return np.nan
        B1=0.6961663; C1=0.0684043**2; B2=0.4079426; C2=0.1162414**2; B3=0.8974794; C3=9.896161**2
        return sellmeier_calc(l_um_sq, B1, C1, B2, C2, B3, C3)
    elif substrate_id == 1:
        min_wl = 400.0
        if wavelength_nm < min_wl: return np.nan
        B1=1.03961212; C1=0.00600069867; B2=0.231792344; C2=0.0200179144; B3=1.01046945; C3=103.560653
        return sellmeier_calc(l_um_sq, B1, C1, B2, C2, B3, C3)
    elif substrate_id == 2:
        min_wl = 360.0
        if wavelength_nm < min_wl: return np.nan
        B1=0.90963095; C1=0.0047563071; B2=0.37290409; C2=0.01621977; B3=0.92110613; C3=105.77911
        return sellmeier_calc(l_um_sq, B1, C1, B2, C2, B3, C3)
    elif substrate_id == 3:
        min_wl = 230.0
        if wavelength_nm < min_wl: return np.nan
        B1 = 2.003059; C1 = 0.011694
        B2 = 0.360392; C2 = 1000.0
        B3 = 0.0;     C3 = 1.0
        return sellmeier_calc(l_um_sq, B1, C1, B2, C2, B3, C3)
    elif substrate_id == 4:
        min_wl = 400.0
        if wavelength_nm < min_wl: return np.nan
        B1=0.90110328; C1=0.0045578115; B2=0.39734436; C2=0.016601149; B3=0.94615601; C3=111.88593
        return sellmeier_calc(l_um_sq, B1, C1, B2, C2, B3, C3)
    else:
        min_wl = 230.0
        if wavelength_nm < min_wl: return np.nan
        B1=0.6961663; C1=0.0684043**2; B2=0.4079426; C2=0.1162414**2; B3=0.8974794; C3=9.896161**2
        return sellmeier_calc(l_um_sq, B1, C1, B2, C2, B3, C3)

@numba.jit(nopython=True, cache=True)
def calculate_monolayer_lambda(l_val, nMono, thickness_nm, nSub_val):
    if not np.isfinite(nSub_val) or nSub_val < 1e-6 : return np.nan, np.nan, np.nan
    n0 = 1.0
    eta_inc_fwd = np.complex128(n0 + 0j); eta_sub_fwd = np.complex128(nSub_val + 0j)
    eta_inc_rev = eta_sub_fwd; eta_sub_rev = eta_inc_fwd
    real_eta_inc_fwd = n0; real_eta_sub_fwd = nSub_val
    real_eta_inc_rev = nSub_val; real_eta_sub_rev = n0

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

    eta1 = nMono; phi1 = (2.0 * PI / l_val) * eta1 * thickness_nm; cos_phi1 = np.cos(phi1); sin_phi1 = np.sin(phi1)
    M1 = np.zeros((2, 2), dtype=np.complex128); M1[0, 0] = cos_phi1; M1[1, 1] = cos_phi1
    if abs(eta1) > SMALL_EPSILON: M1[0, 1] = (1j / eta1) * sin_phi1
    else: M1[0, 1] = np.complex128(0 + 1j*HUGE_PENALTY) if abs(sin_phi1) > SMALL_EPSILON else 0j
    M1[1, 0] = 1j * eta1 * sin_phi1

    M_fwd = M1; M_rev = M1

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

    denom_global_fwd = 1.0 - Ra_minus * Rb_plus; denom_global_bwd = 1.0 - Ra_minus * Rb_minus
    if abs(denom_global_fwd) < SMALL_EPSILON: denom_global_fwd = SMALL_EPSILON
    if abs(denom_global_bwd) < SMALL_EPSILON: denom_global_bwd = SMALL_EPSILON

    R_global_calc = Ra_plus + (Ta_plus * Rb_plus * Ta_minus) / denom_global_fwd
    T_global_calc = (Ta_plus * Tb_plus) / denom_global_fwd
    R_prime_global_calc = Rb_plus + (Tb_plus * Ra_minus * Tb_minus) / denom_global_bwd

    Rs_calc = max(0.0, min(1.0, R_global_calc.real)); Ts_calc = max(0.0, min(1.0, T_global_calc.real)); Rs_prime_calc = max(0.0, min(1.0, R_prime_global_calc.real))

    if not np.isfinite(Rs_calc): Rs_calc = np.nan
    if not np.isfinite(Ts_calc): Ts_calc = np.nan
    if not np.isfinite(Rs_prime_calc): Rs_prime_calc = np.nan

    return Rs_calc, Ts_calc, Rs_prime_calc

@numba.jit(nopython=True, cache=True)
def calculate_total_error_numba(l_array, nSub_array, target_value_array,
                                 weights_array, target_type_flag,
                                 current_thickness_nm,
                                 n_calc_array, k_calc_array,
                                 n_min_bound, n_max_bound, k_min_bound, k_max_bound):
    total_sq_error = 0.0
    points_calculated = 0
    for i in range(len(l_array)):
        l_val = l_array[i]
        nSub_val = nSub_array[i]
        if not np.isfinite(nSub_val): continue

        n_calc = n_calc_array[i]
        k_calc = k_calc_array[i]

        n_calc = max(1.0, n_calc)
        k_calc = max(0.0, k_calc)

        if not (n_min_bound <= n_calc <= n_max_bound): return HUGE_PENALTY
        if not (k_min_bound <= k_calc <= k_max_bound): return HUGE_PENALTY
        if not (np.isfinite(n_calc) and np.isfinite(k_calc)): return HUGE_PENALTY

        nMono_complex_val = n_calc - 1j * k_calc

        _, Ts_stack, _ = calculate_monolayer_lambda(l_val, nMono_complex_val, current_thickness_nm, nSub_val)
        if not np.isfinite(Ts_stack): continue

        calculated_value = np.nan
        if target_type_flag == 0:
            _, Ts_sub, _ = calculate_monolayer_lambda(l_val, 1.0 + 0j, 0.0, nSub_val)
            if not np.isfinite(Ts_sub): continue

            if Ts_sub > SMALL_EPSILON:
                T_norm_calc = Ts_stack / Ts_sub
            else:
                T_norm_calc = 0.0 if abs(Ts_stack) < SMALL_EPSILON else HUGE_PENALTY

            if not np.isfinite(T_norm_calc): return HUGE_PENALTY
            calculated_value = max(0.0, min(2.0, T_norm_calc))

        elif target_type_flag == 1:
            calculated_value = max(0.0, min(1.0, Ts_stack))
        else:
            return HUGE_PENALTY

        if np.isfinite(target_value_array[i]) and np.isfinite(calculated_value):
            error_i = (calculated_value - target_value_array[i])**2
            total_sq_error += error_i * weights_array[i]
            points_calculated += weights_array[i]
        elif not np.isfinite(target_value_array[i]):
            pass
        else:
            return HUGE_PENALTY

    if points_calculated <= SMALL_EPSILON:
        return HUGE_PENALTY
    return total_sq_error / points_calculated

def objective_func_spline_fixed_knots(p, num_knots_n, num_knots_k, l_array, nSub_array, target_value_array, weights_array, target_type_flag, fixed_n_knot_lambdas, fixed_k_knot_lambdas):
    expected_len = 1 + num_knots_n + num_knots_k
    if len(p) != expected_len:
        return HUGE_PENALTY

    current_thickness_nm = p[0]
    idx_start = 1
    n_knot_values = p[idx_start : idx_start + num_knots_n]; idx_start += num_knots_n
    log_k_knot_values = p[idx_start : idx_start + num_knots_k]

    if current_thickness_nm < 0: return HUGE_PENALTY
    if np.any(n_knot_values < N_KNOT_VALUE_BOUNDS[0] - SMALL_EPSILON) or \
       np.any(n_knot_values > N_KNOT_VALUE_BOUNDS[1] + SMALL_EPSILON) or \
       np.any(log_k_knot_values < LOG_K_KNOT_VALUE_BOUNDS[0] - SMALL_EPSILON) or \
       np.any(log_k_knot_values > LOG_K_KNOT_VALUE_BOUNDS[1] + SMALL_EPSILON):
        return HUGE_PENALTY

    try:
        if num_knots_n < 2 or num_knots_k < 2 : return HUGE_PENALTY

        n_spline = CubicSpline(fixed_n_knot_lambdas, n_knot_values, bc_type='natural', extrapolate=True)
        log_k_spline = CubicSpline(fixed_k_knot_lambdas, log_k_knot_values, bc_type='natural', extrapolate=True)

        n_calc_array = n_spline(l_array)
        k_calc_array = np.exp(log_k_spline(l_array))

        n_min_req = N_KNOT_VALUE_BOUNDS[0]
        n_max_req = N_KNOT_VALUE_BOUNDS[1]
        k_min_req = math.exp(LOG_K_KNOT_VALUE_BOUNDS[0])
        k_max_req = math.exp(LOG_K_KNOT_VALUE_BOUNDS[1])

        mse = calculate_total_error_numba(l_array, nSub_array, target_value_array,
                                           weights_array, target_type_flag,
                                           current_thickness_nm,
                                           n_calc_array, k_calc_array,
                                           n_min_req, n_max_req, k_min_req, k_max_req)
        return mse

    except ValueError:
        return HUGE_PENALTY
    except Exception:
        return HUGE_PENALTY

def add_log_message(message_type, message):
    if 'log_messages' not in st.session_state: st.session_state.log_messages = []
    st.session_state.log_messages.append((message_type, message))
    if message_type == "info": logger.info(message)
    elif message_type == "warning": logger.warning(message)
    elif message_type == "error": logger.error(message)

def display_log():
    if 'log_messages' in st.session_state and st.session_state.log_messages:
        with st.expander("Show Session Log", expanded=False):
            log_container = st.container()
            for msg_type, msg in reversed(st.session_state.log_messages):
                if msg_type == "info": log_container.info(msg, icon="ℹ️")
                elif msg_type == "warning": log_container.warning(msg, icon="⚠️")
                elif msg_type == "error": log_container.error(msg, icon="🚨")
                else: log_container.text(msg)

def reset_log():
    st.session_state.log_messages = []

def clear_results():
    st.session_state.optim_results = None
    st.session_state.fig_compare_results = None
    st.session_state.fig_nk_results = None

def plot_target_only(target_data_to_plot, target_filename_base):
    if target_data_to_plot is None or 'lambda' not in target_data_to_plot or len(target_data_to_plot['lambda']) == 0:
        add_log_message("warning", "No target data available to plot.")
        return None

    target_type = target_data_to_plot.get('target_type', 'T_norm')
    target_values = target_data_to_plot.get('target_value', None)
    target_l = target_data_to_plot['lambda']

    if target_values is None:
        add_log_message("warning", "Target values are missing in the data.")
        return None

    fig_target, ax_target = plt.subplots(1, 1, figsize=(8, 6));
    plt.style.use('seaborn-v0_8-whitegrid')
    short_filename = target_filename_base if target_filename_base else "Target"
    if target_type == 'T_norm':
        plot_label = 'Target T Norm (%)'; y_label = 'Normalized Transmission (%)'; title_suffix = "T Norm (%)"; y_lim_top = 110
    else:
        plot_label = 'Target T (%)'; y_label = 'Transmission (%)'; title_suffix = "T Sample (%)"; y_lim_top = 105

    ax_target.set_title(f"Target Data ({short_filename}) - {title_suffix}")
    ax_target.set_xlabel('λ (nm)');
    ax_target.minorticks_on();
    ax_target.grid(True, which='major', linestyle='-', linewidth='0.5', color='gray')
    ax_target.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgray')

    valid_mask = np.isfinite(target_values) & np.isfinite(target_l)
    if np.any(valid_mask):
        ax_target.plot(target_l[valid_mask], target_values[valid_mask] * 100.0, '.', markersize=5, color='red', linestyle='none', label=plot_label)
        ax_target.set_ylabel(y_label)
        min_y_data = np.min(target_values[valid_mask] * 100.0)
        max_y_data = np.max(target_values[valid_mask] * 100.0)
        y_padding = max(5, (max_y_data - min_y_data) * 0.05) if max_y_data > min_y_data else 5
        ax_target.set_ylim(bottom=max(-5, min_y_data - y_padding), top=min(y_lim_top + 10 , max_y_data + y_padding))
    else:
        add_log_message("warning", "No valid (finite) target data points to plot.")
        ax_target.set_ylabel('No Valid Data')
        ax_target.set_ylim(bottom=-5, top=y_lim_top)

    lambda_min_plot = st.session_state.get('lambda_min_file', np.nanmin(target_l[valid_mask]) if np.any(valid_mask) else 0)
    lambda_max_plot = st.session_state.get('lambda_max_file', np.nanmax(target_l[valid_mask]) if np.any(valid_mask) else 1000)

    if lambda_min_plot < lambda_max_plot: ax_target.set_xlim(lambda_min_plot, lambda_max_plot)

    ax_target.legend(fontsize='small')
    plt.tight_layout()
    return fig_target

def draw_schema_matplotlib(_target_type, _substrate_name):
    fig, ax = plt.subplots(figsize=(5.5, 1.5))
    fig.patch.set_alpha(0)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis('off')

    color_Mono = "#D6EAF8"; color_Sub = "#EAECEE"; outline_color = "#5D6D7E"
    layer_font_size = 7; medium_font_size = 7
    layer_font = {'size': layer_font_size, 'color': outline_color}
    medium_font = {'size': medium_font_size, 'weight': 'bold', 'color': outline_color}

    x_left = 10; stack_width = 25; mono_h = 8; sub_h = 8; y_sub_base = 5

    ax.add_patch(plt.Rectangle((x_left, y_sub_base + sub_h), stack_width, mono_h, facecolor=color_Mono, edgecolor=outline_color, linewidth=0.5))
    ax.text(x_left + stack_width/2, y_sub_base + sub_h + mono_h/2, "Monolayer", ha='center', va='center', fontdict=layer_font)
    ax.add_patch(plt.Rectangle((x_left, y_sub_base), stack_width, sub_h, facecolor=color_Sub, edgecolor=outline_color, linewidth=0.5))
    ax.text(x_left + stack_width/2, y_sub_base + sub_h/2, f"Sub ({_substrate_name})", ha='center', va='center', fontdict=layer_font)
    ax.text(x_left + stack_width/2, y_sub_base + sub_h + mono_h + 3, "Air (n≈1)", ha='center', va='top', fontdict=medium_font)
    ax.text(x_left + stack_width/2, y_sub_base - 3, "Air", ha='center', va='bottom', fontdict=medium_font)

    arrow_x = x_left + stack_width/2; y_arrow_start = 28; y_arrow_end = 2
    ax.arrow(arrow_x, y_arrow_start, 0, y_arrow_end - y_arrow_start, head_width=3, head_length=2, fc='darkred', ec='darkred', length_includes_head=True, width=0.5)
    label_text = "T_sample" if _target_type == 'T' else "T_norm"
    ax.text(arrow_x + 4, y_arrow_end + 5, label_text, ha='left', va='center', color='darkred', style='italic', size=8)

    if _target_type == 'T_norm':
        x_right = 100 - 10 - stack_width
        ax.add_patch(plt.Rectangle((x_right, y_sub_base), stack_width, sub_h, facecolor=color_Sub, edgecolor=outline_color, linewidth=0.5))
        ax.text(x_right + stack_width/2, y_sub_base + sub_h/2, f"Sub ({_substrate_name})", ha='center', va='center', fontdict=layer_font)
        ax.text(x_right + stack_width/2, y_sub_base + sub_h + 3, "Air (n≈1)", ha='center', va='top', fontdict=medium_font)
        ax.text(x_right + stack_width/2, y_sub_base - 3, "Air", ha='center', va='bottom', fontdict=medium_font)

        arrow_x_right = x_right + stack_width / 2
        ax.arrow(arrow_x_right, y_arrow_start, 0, y_arrow_end - y_arrow_start, head_width=3, head_length=2, fc='darkred', ec='darkred', length_includes_head=True, width=0.5)
        ax.text(arrow_x_right + 4, y_arrow_end + 5, "T_sub", ha='left', va='center', color='darkred', style='italic', size=8)

        x_center = 50
        text_tnorm = f"{label_text} = T_sample / T_sub"
        ax.text(x_center, 15, text_tnorm, ha='center', va='center', style='italic', size=7, color=outline_color)

    plt.tight_layout()
    return fig

@st.cache_resource
def plot_substrate_indices():
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    lambda_nm = np.linspace(200, 2000, 500)

    for idx, name in enumerate(SUBSTRATE_LIST):
        min_wl_sub = get_substrate_min_lambda(name)
        try:
            n_values = np.array([get_n_substrate(idx, l) for l in lambda_nm])
            valid_mask = lambda_nm >= min_wl_sub
            line, = ax.plot(lambda_nm[valid_mask], n_values[valid_mask], label=f"{name} (≥{min_wl_sub:.0f} nm)", linewidth=1.5)
            invalid_mask = lambda_nm < min_wl_sub
            if np.any(invalid_mask):
                n_invalid = np.array([get_n_substrate(idx, l) for l in lambda_nm[invalid_mask]])
                finite_invalid_mask = np.isfinite(n_invalid)
                ax.plot(lambda_nm[invalid_mask][finite_invalid_mask], n_invalid[finite_invalid_mask], linestyle=':', color=line.get_color(), alpha=0.5)
        except Exception as e:
            logger.warning(f"Could not calculate index for {name}. Error: {e}")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Refractive Index (n)")
    ax.set_title("Substrate Refractive Indices (Sellmeier - Dashed below nominal range)")
    ax.grid(True, which='major', linestyle='-', linewidth='0.5', color='gray')
    ax.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgray')
    ax.minorticks_on()
    ax.legend()
    ax.set_xlim(200, 2000)
    plt.tight_layout()
    return fig

def plot_spectra_vs_target(res, target=None, best_params_info=None, model_str_base="Spline Fit", effective_lambda_min=None, effective_lambda_max=None):
    try:
        fig, ax = plt.subplots(1, 1, figsize=(9, 7)); ax_delta = ax.twinx()
        plt.style.use('seaborn-v0_8-whitegrid')

        num_knots_n = best_params_info.get('num_knots_n', '?'); num_knots_k = best_params_info.get('num_knots_k', '?'); knot_distrib = best_params_info.get('knot_distribution', '?')
        model_str = f"{model_str_base} ({num_knots_n}n/{num_knots_k}k Knots, {knot_distrib})"
        target_type = target.get('target_type', 'T_norm') if target else 'T_norm'

        if target_type == 'T_norm':
            comparison_label = 'Normalized T'; target_label_suffix = 'T Norm (%)'; calc_label_suffix = 'T Norm (%)'; y_label = 'Normalized Transmission (%)'; target_value_key = 'target_value'; calc_value_key = 'T_norm_calc'; y_lim_top = 110
        else:
            comparison_label = 'Sample T'; target_label_suffix = 'T (%)'; calc_label_suffix = 'T (%)'; y_label = 'Transmission (%)'; target_value_key = 'target_value'; calc_value_key = 'T_stack_calc'; y_lim_top = 105

        title_base=f'{comparison_label} Comparison ({model_str})'; params_list = []
        if best_params_info and 'thickness_nm' in best_params_info: params_list.append(f"d={best_params_info['thickness_nm']:.2f}nm")
        if best_params_info and 'substrate_name' in best_params_info: params_list.append(f"Sub={best_params_info['substrate_name']}")
        if effective_lambda_min is not None and effective_lambda_max is not None: params_list.append(f"λ=[{effective_lambda_min:.0f}-{effective_lambda_max:.0f}]nm")
        if params_list: title_base += f' - {", ".join(params_list)}'
        title = title_base;
        if 'MSE_Recalculated' in res and res['MSE_Recalculated'] is not None and np.isfinite(res['MSE_Recalculated']): title += f"\nFinal MSE (in Range): {res['MSE_Recalculated']:.3e}"
        elif 'MSE_Optimized' in res and res['MSE_Optimized'] is not None and np.isfinite(res['MSE_Optimized']): title += f"\nOptim. Objective MSE: {res['MSE_Optimized']:.3e}"

        ax.set_title(title, fontsize=10);
        ax.set_xlabel('λ (nm)'); ax.set_ylabel(y_label); ax.minorticks_on(); ax.set_ylim(bottom=-5, top=y_lim_top)
        ax.grid(True, which='major', linestyle='-', linewidth='0.5', color='gray')
        ax.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgray')

        line_calc = None; calc_l = res.get('l'); calc_y = res.get(calc_value_key)
        if calc_l is not None and calc_y is not None:
            valid_calc_mask = np.isfinite(calc_y) & np.isfinite(calc_l)
            plot_mask_calc = valid_calc_mask
            if np.any(plot_mask_calc):
                line_calc, = ax.plot(calc_l[plot_mask_calc], calc_y[plot_mask_calc] * 100.0, label=f'Calc {calc_label_suffix}', linestyle='-', color='darkblue', linewidth=1.5);

        line_tgt = None; target_l_valid, target_y_valid = None, None
        if target is not None and 'lambda' in target and target_value_key in target and len(target['lambda']) > 0:
            target_l_valid_orig = target['lambda']; target_y_valid_fraction = target[target_value_key]
            valid_target_mask = np.isfinite(target_y_valid_fraction) & np.isfinite(target_l_valid_orig)
            final_target_mask = valid_target_mask
            if np.any(final_target_mask):
                target_l_valid = target_l_valid_orig[final_target_mask]; target_y_valid = target_y_valid_fraction[final_target_mask] * 100.0
                line_tgt, = ax.plot(target_l_valid, target_y_valid, 'o', markersize=4, color='red', fillstyle='none', label=f'Target {target_label_suffix}');

        line_delta = None; delta_t_perc = np.full_like(calc_l, np.nan) if calc_l is not None else np.array([])
        if calc_l is not None and calc_y is not None and target_l_valid is not None and target_y_valid is not None and len(target_l_valid) > 1:
            calc_y_perc = calc_y * 100.0
            try:
                target_y_perc_interp = np.interp(calc_l, target_l_valid, target_y_valid, left=np.nan, right=np.nan)
                delta_t_perc_full = calc_y_perc - target_y_perc_interp

                valid_delta_mask = np.isfinite(delta_t_perc_full) & np.isfinite(calc_l)
                plot_mask_delta = (calc_l >= effective_lambda_min) & (calc_l <= effective_lambda_max) & valid_delta_mask

                if np.any(plot_mask_delta):
                    line_delta, = ax_delta.plot(calc_l[plot_mask_delta], delta_t_perc_full[plot_mask_delta], label='ΔT (%) [Calc - Target, Optim. Range]', linestyle=':', color='green', linewidth=1.2, zorder=-5);
                    min_delta = np.min(delta_t_perc_full[plot_mask_delta]); max_delta = np.max(delta_t_perc_full[plot_mask_delta])
                    padding = max(1.0, abs(max_delta - min_delta) * 0.1) if max_delta != min_delta else 1.0
                    ax_delta.set_ylim(min_delta - padding, max_delta + padding); ax_delta.set_ylabel('ΔT (%) [Optim. Range]', color='green'); ax_delta.tick_params(axis='y', labelcolor='green'); ax_delta.grid(True, axis='y', linestyle='-.', linewidth=0.5, color='lightgreen', alpha=0.6)
                else:
                    ax_delta.set_ylabel('ΔT (%) [Optim. Range]', color='green'); ax_delta.tick_params(axis='y', labelcolor='green'); ax_delta.set_yticks([])

            except Exception as e_interp:
                add_log_message("warning", f"Could not interpolate target for delta plot: {e_interp}")
                ax_delta.set_ylabel('ΔT (%) [Optim. Range]', color='green'); ax_delta.tick_params(axis='y', labelcolor='green'); ax_delta.set_yticks([])

        file_lambda_min = st.session_state.get('lambda_min_file')
        file_lambda_max = st.session_state.get('lambda_max_file')
        if file_lambda_min is not None and file_lambda_max is not None and file_lambda_min < file_lambda_max:
            ax.set_xlim(file_lambda_min, file_lambda_max)
        elif effective_lambda_min is not None and effective_lambda_max is not None and effective_lambda_min < effective_lambda_max:
            ax.set_xlim(effective_lambda_min, effective_lambda_max)
        else:
            min_l_plot_fallback = np.nanmin(res.get('l', [300])); max_l_plot_fallback = np.nanmax(res.get('l', [1000]));
            if np.isfinite(min_l_plot_fallback) and np.isfinite(max_l_plot_fallback) and min_l_plot_fallback < max_l_plot_fallback:
                ax.set_xlim(min_l_plot_fallback, max_l_plot_fallback)

        handles1, labels1 = ax.get_legend_handles_labels(); handles2, labels2 = ax_delta.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc='best', fontsize='small')

        percent_good_fit = res.get('percent_good_fit', np.nan); quality_label = res.get('quality_label', 'N/A')
        if np.isfinite(percent_good_fit):
            quality_text = f"Fit Quality (in Optim. Range): {quality_label}\n(<0.25% abs delta): {percent_good_fit:.1f}%"
            ax.text(0.98, 0.02, quality_text, transform=ax.transAxes, fontsize=10, ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', alpha=0.8))

        plt.tight_layout()
        return fig
    except Exception as e_plot:
        add_log_message("error", f"Failed to generate final spectra plot: {e_plot}")
        return None

def plot_nk_final(best_params_info, plot_lambda_array):
    fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
    plt.style.use('seaborn-v0_8-whitegrid')
    ax2 = ax1.twinx()

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

        color1='tab:red'; color2='tab:blue'

        ax1.set_xlabel('λ (nm)'); ax1.set_ylabel("k Coeff.", color=color1);
        ax1.plot(plot_lambda_array, k_plot, color=color1, linestyle='--', linewidth=1.5, label='k (Final)')
        ax1.plot(fixed_k_lambdas, np.exp(log_k_values_opt_final), 's', color=color1, markersize=6, fillstyle='none', label=f'k Knots ({num_k})')
        ax1.tick_params(axis='y', labelcolor=color1); ax1.grid(True,which='major',ls=':',lw=0.5,axis='y',color=color1)

        ax2.set_ylabel('n Index', color=color2)
        ax2.plot(plot_lambda_array, n_plot, color=color2, linestyle='-', linewidth=1.5, label='n (Final)')
        ax2.plot(fixed_n_lambdas, n_values_opt_final, 'o', color=color2, markersize=6, fillstyle='none', label=f'n Knots ({num_n})')
        ax2.tick_params(axis='y', labelcolor=color2)

        ax1.minorticks_on();
        ax1.grid(True,which='major',ls='-',lw=0.5,axis='x')
        ax1.grid(True,which='minor',ls=':',lw=0.5,axis='x')

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

        n_min_req = N_KNOT_VALUE_BOUNDS[0]; n_max_req = N_KNOT_VALUE_BOUNDS[1]
        n_range = n_max_req - n_min_req
        n_low_lim = n_min_req - 0.1 * n_range if n_range > 0 else n_min_req * 0.95
        n_high_lim = n_max_req + 0.1 * n_range if n_range > 0 else n_max_req * 1.05
        min_n_plot_val = np.nanmin(n_plot) if n_plot is not None and np.any(np.isfinite(n_plot)) else n_min_req
        max_n_plot_val = np.nanmax(n_plot) if n_plot is not None and np.any(np.isfinite(n_plot)) else n_max_req
        min_n_ylim = min(n_low_lim, min_n_plot_val * 0.98 if np.isfinite(min_n_plot_val) else n_low_lim)
        max_n_ylim = max(n_high_lim, max_n_plot_val * 1.02 if np.isfinite(max_n_plot_val) else n_high_lim)
        ax2.set_ylim(bottom=min_n_ylim, top=max_n_ylim)

        if lambda_min_eff is not None and lambda_max_eff is not None and lambda_min_eff < lambda_max_eff :
            ax1.set_xlim(lambda_min_eff, lambda_max_eff)

        ax1.set_title('Final Optimized n/k Indices')
        handles1, labels1 = ax1.get_legend_handles_labels(); handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc='best', fontsize='small')
        plt.tight_layout()
        return fig

    except Exception as e:
        add_log_message("error", f"Failed to generate final n/k plot: {e}")
        return None

def log_user_access(timestamp, user_name, user_email):
    file_exists = os.path.isfile(USER_LOG_FILE)
    try:
        with open(USER_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists or os.path.getsize(USER_LOG_FILE) == 0:
                writer.writerow(['Timestamp', 'Name', 'Email'])
            writer.writerow([timestamp, user_name, user_email])
        return True
    except Exception as e:
        logger.error(f"Failed to write to user log file {USER_LOG_FILE}: {e}")
        if 'log_messages' in st.session_state:
            add_log_message("error", f"Failed to write to user log file {USER_LOG_FILE}: {e}")
        return False

def display_user_log():
    st.subheader("User Access Log")
    if os.path.isfile(USER_LOG_FILE):
        try:
            df_log = pd.read_csv(USER_LOG_FILE)
            st.metric("Total Access Records", len(df_log))
            st.dataframe(df_log.iloc[::-1], use_container_width=True)
        except pd.errors.EmptyDataError:
             st.info("User log file is empty.")
        except Exception as e:
            st.error(f"Could not read user log file {USER_LOG_FILE}: {e}")
            add_log_message("error", f"Failed to read user log file: {e}")
    else:
        st.info("User log file not found or is empty.")

default_advanced_params = {
    'num_knots_n': 6, 'num_knots_k': 6, 'use_inv_lambda_sq_distrib': False,
    'pop_size': 20, 'maxiter': 1500, 'tol': 0.001, 'atol': 0.0,
    'mutation_min': 0.5, 'mutation_max': 1.2, 'recombination': 0.8,
    'strategy': 'best1bin', 'polish': True, 'updating': 'deferred', 'workers': -1
}

def init_session_state():
    defaults = {
        'target_data': None,
        'target_filename_base': None,
        'lambda_min_file': None,
        'lambda_max_file': None,
        'log_messages': [],
        'optim_results': None,
        'config_lambda_min': "---",
        'config_lambda_max': "---",
        'thickness_min': 300.0,
        'thickness_max': 600.0,
        'substrate_choice': SUBSTRATE_LIST[0],
        'target_type': "T_norm",
        'last_loaded_source': None,
        'info_submitted': False,
        'user_name': "",
        'user_email': "",
        'fig_substrate_plot': None,
        'fig_target_plot': None,
        'fig_compare_results': None,
        'fig_nk_results': None,
        'advanced_optim_params': default_advanced_params.copy()
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

st.set_page_config(page_title="Monolayer Optimizer", layout="wide")

init_session_state()

if not st.session_state.info_submitted:
    st.header("Welcome!")
    st.write("Please provide your details (optional) to continue.")
    st.info("Privacy Notice: Your details (if provided) are logged for usage tracking.", icon="ℹ️")

    with st.form("info_form"):
        name_input = st.text_input("Your name (optional)")
        email_input = st.text_input("Your email address (optional)")
        submitted = st.form_submit_button("Access Application")

        if submitted:
            st.session_state.user_name = name_input.strip() if name_input else ""
            st.session_state.user_email = email_input.strip() if email_input else ""
            st.session_state.info_submitted = True
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_user_access(now_str, st.session_state.user_name, st.session_state.user_email)
            st.rerun()
else:
    st.title("Monolayer Optical Properties Optimizer")

    tab1, tab2, tab3 = st.tabs(["Configuration", "Target Data", "Run & Results"])

    with tab1:
        st.header("Optimization Configuration")
        col_cfg1, col_cfg2 = st.columns(2)

        with col_cfg1:
            st.subheader("Substrate & Wavelength Range")
            prev_substrate = st.session_state.substrate_choice
            st.session_state.substrate_choice = st.selectbox(
                "Substrate Material:", SUBSTRATE_LIST,
                index=SUBSTRATE_LIST.index(st.session_state.substrate_choice),
                key="config_substrate_select"
            )

            if st.session_state.substrate_choice != prev_substrate and st.session_state.target_data is not None:
                substrate_min_wl = get_substrate_min_lambda(st.session_state.substrate_choice)
                if st.session_state.lambda_min_file is not None:
                    new_config_min = max(st.session_state.lambda_min_file, substrate_min_wl)
                    try:
                        current_max_str = st.session_state.config_lambda_max
                        current_max = float(current_max_str) if current_max_str != "---" else np.inf
                        if new_config_min < current_max:
                            st.session_state.config_lambda_min = f"{new_config_min:.1f}"
                            add_log_message("info", f"Substrate changed. Default Min λ updated to {st.session_state.config_lambda_min} nm.")
                        else:
                             add_log_message("warning", f"Substrate changed. New suggested Min λ ({new_config_min:.1f}) >= Max λ ({current_max_str}). Manual adjustment needed.")
                    except ValueError:
                         st.session_state.config_lambda_min = f"{new_config_min:.1f}"
                         add_log_message("info", f"Substrate changed. Default Min λ updated to {st.session_state.config_lambda_min} nm.")
                    st.rerun()

            st.text("Optimization Lambda Range (nm):")
            sub_limit = get_substrate_min_lambda(st.session_state.substrate_choice)
            file_min = st.session_state.get('lambda_min_file', None)
            file_max = st.session_state.get('lambda_max_file', None)
            help_min = f"≥ Substrate limit ({sub_limit:.1f}) " + (f"and ≥ File min ({file_min:.1f})" if file_min is not None else "")
            help_max = f"≤ File max ({file_max:.1f})" if file_max is not None else ""

            subcol_lam1, subcol_lam2 = st.columns(2)
            with subcol_lam1:
                st.session_state.config_lambda_min = st.text_input("Min λ", value=st.session_state.config_lambda_min, key="cfg_lam_min_input", help=help_min)
            with subcol_lam2:
                st.session_state.config_lambda_max = st.text_input("Max λ", value=st.session_state.config_lambda_max, key="cfg_lam_max_input", help=help_max)

            try:
                lmin_str = st.session_state.config_lambda_min
                lmax_str = st.session_state.config_lambda_max
                lmin = float(lmin_str) if lmin_str != "---" else -1
                lmax = float(lmax_str) if lmax_str != "---" else -1
                if lmin != -1 and lmin < sub_limit: st.warning(f"Min λ ({lmin:.1f}) < Substrate limit ({sub_limit:.1f})!", icon="⚠️")
                if lmin != -1 and lmax != -1 and lmin >= lmax: st.warning("Min λ ≥ Max λ!", icon="⚠️")
                if file_min is not None and lmin != -1 and lmin < file_min: st.warning(f"Min λ ({lmin:.1f}) < File min ({file_min:.1f})!", icon="⚠️")
                if file_max is not None and lmax != -1 and lmax > file_max: st.warning(f"Max λ ({lmax:.1f}) > File max ({file_max:.1f})!", icon="⚠️")
            except ValueError:
                if lmin_str != "---" or lmax_str != "---": st.warning("Invalid number format for λ Min/Max.", icon="⚠️")

            if st.button("Plot Substrate Indices", key="plot_sub_button"):
                st.session_state.fig_substrate_plot = plot_substrate_indices()
                st.rerun()

            if st.session_state.fig_substrate_plot:
                 st.pyplot(st.session_state.fig_substrate_plot)
                 if st.button("Clear Substrate Plot", key="clear_sub_plot"):
                     st.session_state.fig_substrate_plot = None;
                     st.rerun()

            st.subheader("Thickness Range")
            subcol_th1, subcol_th2 = st.columns(2)
            with subcol_th1: st.session_state.thickness_min = st.number_input("Min Thick (nm)", min_value=0.0, value=st.session_state.thickness_min, step=10.0, format="%.1f", key="cfg_thick_min")
            with subcol_th2: st.session_state.thickness_max = st.number_input("Max Thick (nm)", min_value=max(0.1, st.session_state.thickness_min + 0.1), value=st.session_state.thickness_max, step=10.0, format="%.1f", key="cfg_thick_max")

        with col_cfg2:
            with st.expander("Advanced Optimization Settings", expanded=False):
                adv_params = st.session_state.advanced_optim_params
                adv_params['num_knots_n'] = st.number_input("n Spline Knots", min_value=2, value=adv_params['num_knots_n'], step=1, key="adv_nknots")
                adv_params['num_knots_k'] = st.number_input("k Spline Knots", min_value=2, value=adv_params['num_knots_k'], step=1, key="adv_kknots")
                adv_params['use_inv_lambda_sq_distrib'] = st.checkbox("Knot Distribution 1/λ² (else 1/λ)", value=adv_params['use_inv_lambda_sq_distrib'], key="adv_distrib")
                st.markdown("---")
                st.subheader("Differential Evolution Parameters")
                adv_params['pop_size'] = st.number_input("Population Size", min_value=5, value=adv_params['pop_size'], step=5, key="adv_pop")
                adv_params['maxiter'] = st.number_input("Max Iterations", min_value=1, value=adv_params['maxiter'], step=100, key="adv_iter")
                scol1, scol2 = st.columns(2)
                with scol1: adv_params['tol'] = st.number_input("Relative Tol", min_value=0.0, value=adv_params['tol'], format="%.4f", step=0.001, key="adv_tol")
                with scol2: adv_params['atol'] = st.number_input("Absolute Tol", min_value=0.0, value=adv_params['atol'], format="%.4f", step=0.001, key="adv_atol")
                scol3, scol4 = st.columns(2)
                with scol3: adv_params['mutation_min'] = st.number_input("Mutation Min", min_value=0.0, max_value=2.0, value=adv_params['mutation_min'], step=0.1, key="adv_mutmin")
                with scol4: adv_params['mutation_max'] = st.number_input("Mutation Max", min_value=adv_params['mutation_min'], max_value=2.0, value=adv_params['mutation_max'], step=0.1, key="adv_mutmax")
                adv_params['recombination'] = st.slider("Recombination", 0.0, 1.0, value=adv_params['recombination'], step=0.1, key="adv_recomb")
                adv_params['polish'] = st.checkbox("Polish final solution", value=adv_params['polish'], key="adv_polish")
                strategy_options = ['best1bin', 'best1exp', 'rand1exp', 'randtobest1exp', 'currenttobest1exp', 'best2exp', 'rand2exp', 'randtobest1bin', 'currenttobest1bin', 'best2bin', 'rand2bin', 'rand1bin']
                adv_params['strategy'] = st.selectbox("DE Strategy", options=strategy_options, index=strategy_options.index(adv_params['strategy']), key="adv_strat")
                updating_options = ['immediate', 'deferred']
                adv_params['updating'] = st.selectbox("Updating Mode", options=updating_options, index=updating_options.index(adv_params['updating']), key="adv_update")
                adv_params['workers'] = st.number_input("Parallel Workers (-1=Auto)", min_value=-1, value=adv_params['workers'], step=1, key="adv_work", help="Uses CPU cores. Set to 1 if errors occur.")

        st.divider()
        if st.button("🔄 Reset Configuration Parameters", key="reset_cfg"):
            st.session_state.config_lambda_min = "---"
            st.session_state.config_lambda_max = "---"
            st.session_state.thickness_min = 300.0
            st.session_state.thickness_max = 600.0
            st.session_state.advanced_optim_params = default_advanced_params.copy()
            if st.session_state.target_data is not None:
                 substrate_min_wl_reset = get_substrate_min_lambda(st.session_state.substrate_choice)
                 initial_config_min_reset = max(st.session_state.lambda_min_file, substrate_min_wl_reset)
                 if initial_config_min_reset < st.session_state.lambda_max_file:
                     st.session_state.config_lambda_min = f"{initial_config_min_reset:.1f}"
                     st.session_state.config_lambda_max = f"{st.session_state.lambda_max_file:.1f}"
            clear_results()
            add_log_message("info", "Configuration parameters reset.")
            st.rerun()

    with tab2:
        st.header("Target Data Input")
        col_tgt1, col_tgt2 = st.columns([0.6, 0.4])

        with col_tgt1:
            st.subheader("Select Type & Load File")
            st.session_state.target_type = st.radio(
                "Select Target Type:",
                options=["T_norm", "T"],
                format_func=lambda x: "T Norm (%) = T_sample / T_sub" if x == "T_norm" else "T Sample (%)",
                index=["T_norm", "T"].index(st.session_state.target_type),
                horizontal=True,
                key="target_type_selector"
            )

            uploaded_file = st.file_uploader(
                "Upload Target File (.csv) or use default:",
                type=["csv"],
                accept_multiple_files=False,
                key="target_file_uploader"
            )

            data_source_to_load = None
            source_name = None
            is_default_file = False

            if uploaded_file is not None:
                if uploaded_file.name != st.session_state.get('last_loaded_source', None):
                    data_source_to_load = uploaded_file
                    source_name = uploaded_file.name
                    st.session_state.last_loaded_source = source_name
                    add_log_message("info", f"User uploaded file: {source_name}")
                    reset_log()
                    clear_results()
            elif st.session_state.target_data is None:
                add_log_message("info", f"Checking for default file: {DEFAULT_EXAMPLE_FILE}")
                if os.path.exists(DEFAULT_EXAMPLE_FILE):
                    if DEFAULT_EXAMPLE_FILE != st.session_state.get('last_loaded_source', None):
                        data_source_to_load = DEFAULT_EXAMPLE_FILE
                        source_name = DEFAULT_EXAMPLE_FILE
                        st.session_state.last_loaded_source = source_name
                        is_default_file = True
                        reset_log()
                        clear_results()
                else:
                    if st.session_state.last_loaded_source != "DEFAULT_NOT_FOUND":
                        add_log_message("error", f"Default file '{DEFAULT_EXAMPLE_FILE}' not found.")
                        st.warning(f"Default file '{DEFAULT_EXAMPLE_FILE}' not found. Please upload a file.", icon="⚠️")
                        st.session_state.last_loaded_source = "DEFAULT_NOT_FOUND"

            if data_source_to_load is not None:
                st.session_state.target_data = None
                st.session_state.fig_target_plot = None
                try:
                    file_extension = os.path.splitext(source_name)[1].lower()
                    if file_extension != ".csv": raise ValueError(f"Unsupported file extension: '{file_extension}'. Please select a .csv file.")
                    add_log_message("info", f"Parsing {source_name}...")
                    df = None; error_messages = []
                    parse_attempts = [
                        {'delimiter': ',', 'decimal': '.', 'encoding': 'utf-8', 'msg': "Comma/Dot/UTF-8"},
                        {'delimiter': ';', 'decimal': '.', 'encoding': 'utf-8', 'msg': "Semicolon/Dot/UTF-8"},
                        {'delimiter': '\t', 'decimal': '.', 'encoding': 'utf-8', 'msg': "Tab/Dot/UTF-8"},
                        {'delimiter': ',', 'decimal': ',', 'encoding': 'latin-1', 'msg': "Comma/Comma/Latin-1"},
                        {'delimiter': ';', 'decimal': ',', 'encoding': 'latin-1', 'msg': "Semicolon/Comma/Latin-1"},
                    ]
                    for attempt in parse_attempts:
                        try:
                            if hasattr(data_source_to_load, 'seek'): data_source_to_load.seek(0)
                            df_attempt = pd.read_csv( data_source_to_load, delimiter=attempt['delimiter'], decimal=attempt['decimal'], skiprows=1, usecols=[0, 1], names=['lambda', 'target_value'], encoding=attempt['encoding'], skipinitialspace=True, on_bad_lines='warn' )
                            if df_attempt is not None and not df_attempt.empty and 'lambda' in df_attempt.columns and 'target_value' in df_attempt.columns:
                                lam_test = pd.to_numeric(df_attempt['lambda'], errors='coerce')
                                tgt_test = pd.to_numeric(df_attempt['target_value'], errors='coerce')
                                if not lam_test.isnull().all() and not tgt_test.isnull().all():
                                    df = df_attempt
                                    add_log_message("info", f"Successfully parsed with: {attempt['msg']}")
                                    break
                        except Exception as e_parse: error_messages.append(f"Attempt ({attempt['msg']}) failed: {e_parse}")
                    if df is None: raise ValueError(f"Could not parse CSV '{source_name}'. Check format. Errors: {'; '.join(error_messages)}")
                    if df.empty: raise ValueError("CSV loaded but appears empty.")
                    if 'lambda' not in df.columns or 'target_value' not in df.columns: raise ValueError("Missing 'lambda' or 'target_value' columns.")

                    df['lambda'] = pd.to_numeric(df['lambda'], errors='coerce'); df['target_value'] = pd.to_numeric(df['target_value'], errors='coerce')
                    initial_rows = len(df); df.dropna(subset=['lambda', 'target_value'], inplace=True); rows_after_na = len(df)
                    if rows_after_na < initial_rows: add_log_message("warning", f"{initial_rows - rows_after_na} rows dropped (non-numeric).")
                    if df.empty: raise ValueError("No valid numeric data rows found.")
                    data = df.to_numpy(); data = data[data[:, 0].argsort()]; lam = data[:, 0]; target_value_raw = data[:, 1]
                    if len(lam) < 3: raise ValueError("Not enough valid data rows (< 3).")
                    if np.min(lam) <= 0: raise ValueError("Wavelengths (λ) must be > 0.")

                    valid_targets = target_value_raw[np.isfinite(target_value_raw)]
                    if len(valid_targets) > 0 and np.nanmax(valid_targets) > 5:
                        target_value_final = target_value_raw / 100.0
                        add_log_message("info", f"Target values > 5 detected, dividing by 100.")
                    else: target_value_final = target_value_raw

                    st.session_state.target_data = {'lambda': lam, 'target_value': target_value_final, 'target_type': st.session_state.target_type}
                    st.session_state.lambda_min_file = np.min(lam); st.session_state.lambda_max_file = np.max(lam)
                    st.session_state.target_filename_base = source_name
                    st.session_state.fig_target_plot = plot_target_only(st.session_state.target_data, st.session_state.target_filename_base)

                    current_substrate = st.session_state.substrate_choice; substrate_min_wl = get_substrate_min_lambda(current_substrate); initial_config_min = max(st.session_state.lambda_min_file, substrate_min_wl)
                    if initial_config_min >= st.session_state.lambda_max_file:
                         st.session_state.config_lambda_min = "---"; st.session_state.config_lambda_max = "---"
                         add_log_message("warning", f"Min allowed λ ({initial_config_min:.1f}) >= Max file λ ({st.session_state.lambda_max_file:.1f}). Cannot set default range.")
                    else:
                         st.session_state.config_lambda_min = f"{initial_config_min:.1f}"; st.session_state.config_lambda_max = f"{st.session_state.lambda_max_file:.1f}";
                         add_log_message("info", f"Default optim range set to: [{st.session_state.config_lambda_min}, {st.session_state.config_lambda_max}] nm")

                    log_source = "Default file" if is_default_file else "Uploaded file"
                    add_log_message("info", f"{log_source} '{st.session_state.target_filename_base}' loaded ({rows_after_na} valid rows).")
                    st.rerun()

                except (ValueError, ImportError, Exception) as e:
                    st.error(f"Error reading/processing '{source_name}': {e}", icon="🚨"); add_log_message("error", f"ERROR loading '{source_name}': {e}")
                    st.session_state.target_data = None; st.session_state.lambda_min_file = None; st.session_state.lambda_max_file = None;
                    st.session_state.target_filename_base = "Error"; st.session_state.config_lambda_min = "---"; st.session_state.config_lambda_max = "---"; st.session_state.fig_target_plot = None
                    st.session_state.last_loaded_source = f"ERROR_{source_name}"
                    clear_results()
                    st.rerun()

            if st.session_state.target_filename_base:
                 if st.session_state.target_filename_base == "Error": pass
                 elif st.session_state.target_data is not None:
                     file_source_msg = "(Default)" if st.session_state.last_loaded_source == DEFAULT_EXAMPLE_FILE else "(Uploaded)"
                     st.success(f"Loaded: {st.session_state.target_filename_base} {file_source_msg} (Type: {st.session_state.target_type})", icon="✅")
                     if st.session_state.fig_target_plot:
                         st.pyplot(st.session_state.fig_target_plot)
                     else:
                         st.warning("Target plot not available.", icon="⚠️")
            elif st.session_state.last_loaded_source != "DEFAULT_NOT_FOUND":
                 st.info("Upload a CSV target file or ensure 'example.csv' is present.")

        with col_tgt2:
            st.subheader("Measurement Schema")
            fig_schema = draw_schema_matplotlib(st.session_state.target_type, st.session_state.substrate_choice)
            st.pyplot(fig_schema)

    with tab3:
        st.header("Run Optimization & View Results")

        run_button = st.button("▶ Run Optimization", type="primary", use_container_width=True, disabled=(st.session_state.target_data is None))
        status_placeholder = st.empty()

        if run_button:
            reset_log(); clear_results()
            add_log_message("info", "="*20 + " Starting Optimization " + "="*20)
            valid_params = True
            try:
                thickness_min = float(st.session_state.thickness_min); thickness_max = float(st.session_state.thickness_max)
                if thickness_min > thickness_max or thickness_min < 0: raise ValueError("Invalid Thickness bounds.")
                lambda_min_str = st.session_state.config_lambda_min; lambda_max_str = st.session_state.config_lambda_max
                if lambda_min_str == "---" or lambda_max_str == "---": raise ValueError("Lambda Min/Max not set.")
                effective_lambda_min = float(lambda_min_str); effective_lambda_max = float(lambda_max_str)
                if effective_lambda_min <= 0 or effective_lambda_max <= 0: raise ValueError("Lambda Min/Max must be positive.")
                if effective_lambda_min >= effective_lambda_max: raise ValueError("Lambda Min >= Max Lambda.")
                if st.session_state.lambda_min_file is None: raise ValueError("Target file not loaded.")
                if effective_lambda_min < st.session_state.lambda_min_file - SMALL_EPSILON or effective_lambda_max > st.session_state.lambda_max_file + SMALL_EPSILON: raise ValueError(f"Optim λ range [{effective_lambda_min:.1f}, {effective_lambda_max:.1f}] outside file range [{st.session_state.lambda_min_file:.1f}, {st.session_state.lambda_max_file:.1f}].")
                selected_substrate = st.session_state.substrate_choice; substrate_min_limit = get_substrate_min_lambda(selected_substrate)
                if effective_lambda_min < substrate_min_limit:
                    add_log_message("warning", f"Min λ ({effective_lambda_min:.1f}) < Substrate limit ({substrate_min_limit:.1f}). Adjusting.")
                    effective_lambda_min = substrate_min_limit; st.session_state.config_lambda_min = f"{effective_lambda_min:.1f}"
                    if effective_lambda_min >= effective_lambda_max: raise ValueError(f"Adjusted Min λ ({effective_lambda_min:.1f}) >= Max λ ({effective_lambda_max:.1f}).")
                    st.rerun()
                add_log_message("info", f"Optim range: [{effective_lambda_min:.1f}, {effective_lambda_max:.1f}] nm")
                current_target_type = st.session_state.target_type; target_type_flag = 0 if current_target_type == 'T_norm' else 1
                substrate_id = SUBSTRATE_LIST.index(selected_substrate)
                adv_params = st.session_state.advanced_optim_params; num_knots_n = adv_params['num_knots_n']; num_knots_k = adv_params['num_knots_k']; use_inv_lambda_sq_distrib = adv_params['use_inv_lambda_sq_distrib']
                if num_knots_n < 2 or num_knots_k < 2: raise ValueError("Need >= 2 n/k knots.")

            except (ValueError, TypeError, KeyError) as e_param: st.error(f"Parameter Error: {e_param}", icon="🚨"); add_log_message("error", f"Parameter Error: {e_param}"); valid_params = False
            except Exception as e_unexpected: st.error(f"Setup Error: {e_unexpected}", icon="🚨"); add_log_message("error", f"Unexpected Setup Error: {e_unexpected}"); valid_params = False

            if valid_params:
                current_target_lambda = st.session_state.target_data['lambda']; current_target_value = st.session_state.target_data['target_value']
                lambda_range_mask = (current_target_lambda >= effective_lambda_min) & (current_target_lambda <= effective_lambda_max)
                valid_target_mask_finite = np.isfinite(current_target_value) & np.isfinite(current_target_lambda)
                nSub_target_array_full = np.array([get_n_substrate(substrate_id, l) for l in current_target_lambda])
                valid_substrate_mask = np.isfinite(nSub_target_array_full)
                mask_used_in_optimization = valid_target_mask_finite & lambda_range_mask & valid_substrate_mask

                if not np.any(mask_used_in_optimization):
                    st.error(f"No valid target points in range [{effective_lambda_min:.1f}, {effective_lambda_max:.1f}] with valid substrate index.", icon="🚨"); add_log_message("error", "No valid points for optimization.")
                else:
                    target_lambda_opt = current_target_lambda[mask_used_in_optimization]; target_value_opt = current_target_value[mask_used_in_optimization]; nSub_target_array_opt = nSub_target_array_full[mask_used_in_optimization]
                    weights_array = np.ones_like(target_lambda_opt);
                    add_log_message("info", f"Using {len(target_lambda_opt)} points for optimization.")

                    fixed_n_knot_lambdas = np.array([], dtype=float); fixed_k_knot_lambdas = np.array([], dtype=float); knot_lam_min = effective_lambda_min; knot_lam_max = effective_lambda_max
                    try:
                        if use_inv_lambda_sq_distrib:
                            inv_lambda_sq_min = 1.0 / (knot_lam_max**2); inv_lambda_sq_max = 1.0 / (knot_lam_min**2)
                            if num_knots_n > 0: fixed_n_knot_lambdas = 1.0 / np.sqrt(np.linspace(inv_lambda_sq_min, inv_lambda_sq_max, num_knots_n) + SMALL_EPSILON)
                            if num_knots_k > 0: fixed_k_knot_lambdas = 1.0 / np.sqrt(np.linspace(inv_lambda_sq_min, inv_lambda_sq_max, num_knots_k) + SMALL_EPSILON)
                        else:
                            inv_lambda_min = 1.0 / knot_lam_max; inv_lambda_max = 1.0 / knot_lam_min
                            if num_knots_n > 0: fixed_n_knot_lambdas = 1.0 / (np.linspace(inv_lambda_min, inv_lambda_max, num_knots_n) + SMALL_EPSILON)
                            if num_knots_k > 0: fixed_k_knot_lambdas = 1.0 / (np.linspace(inv_lambda_min, inv_lambda_max, num_knots_k) + SMALL_EPSILON)
                        fixed_n_knot_lambdas = np.clip(np.sort(fixed_n_knot_lambdas), knot_lam_min + SMALL_EPSILON, knot_lam_max - SMALL_EPSILON)
                        fixed_k_knot_lambdas = np.clip(np.sort(fixed_k_knot_lambdas), knot_lam_min + SMALL_EPSILON, knot_lam_max - SMALL_EPSILON)
                        if len(np.unique(fixed_n_knot_lambdas)) < num_knots_n or len(np.unique(fixed_k_knot_lambdas)) < num_knots_k:
                            add_log_message("warning", "Duplicate knots generated. Adjusting slightly.")
                            fixed_n_knot_lambdas = np.linspace(knot_lam_min + SMALL_EPSILON, knot_lam_max - SMALL_EPSILON, num_knots_n); fixed_k_knot_lambdas = np.linspace(knot_lam_min + SMALL_EPSILON, knot_lam_max - SMALL_EPSILON, num_knots_k)
                    except Exception as e_knot: st.error(f"Knot Error: {e_knot}", icon="🚨"); add_log_message("error", f"Knot calculation error: {e_knot}"); valid_params = False

                    if valid_params:
                        parameter_bounds = [(thickness_min, thickness_max)] + \
                                           [N_KNOT_VALUE_BOUNDS] * num_knots_n + \
                                           [LOG_K_KNOT_VALUE_BOUNDS] * num_knots_k
                        fixed_args = (num_knots_n, num_knots_k, target_lambda_opt, nSub_target_array_opt, target_value_opt, weights_array, target_type_flag, fixed_n_knot_lambdas, fixed_k_knot_lambdas)

                        optim_iteration_count = [0]; optim_callback_best_mse = [np.inf]
                        def optimization_callback_simple_log(xk, convergence):
                            optim_iteration_count[0] += 1; display_freq = 5
                            if optim_iteration_count[0] % display_freq == 0 or optim_iteration_count[0] == 1:
                                try:
                                    current_fun = objective_func_spline_fixed_knots(xk, *fixed_args)
                                    if not np.isfinite(current_fun): current_fun = np.inf
                                    if current_fun < optim_callback_best_mse[0]: optim_callback_best_mse[0] = current_fun
                                    mse_val = optim_callback_best_mse[0]
                                    status_placeholder.info(f"Iteration: {optim_iteration_count[0]} | Best MSE: {mse_val:.4e}", icon="⏳")
                                except Exception as e_cb: add_log_message("warning", f"Callback Error iter {optim_iteration_count[0]}: {e_cb}")

                        de_args = { 'func': objective_func_spline_fixed_knots, 'bounds': parameter_bounds, 'args': fixed_args, 'strategy': adv_params['strategy'], 'maxiter': adv_params['maxiter'], 'popsize': adv_params['pop_size'], 'tol': adv_params['tol'], 'atol': adv_params['atol'], 'mutation': (adv_params['mutation_min'], adv_params['mutation_max']), 'recombination': adv_params['recombination'], 'polish': adv_params['polish'], 'updating': adv_params['updating'], 'workers': adv_params['workers'], 'disp': False, 'callback': optimization_callback_simple_log }

                        try:
                            with st.spinner("Optimization running... Please wait."):
                                start_time_opt = time.time();
                                optim_result = scipy.optimize.differential_evolution(**de_args);
                                end_time_opt = time.time()
                            status_placeholder.success(f"Optimization finished in {end_time_opt - start_time_opt:.2f} s.", icon="✅"); add_log_message("info", f"Optimization finished in {end_time_opt - start_time_opt:.2f} s.")

                            if not optim_result.success: add_log_message("warning", f"Optim did not converge: {optim_result.message}"); st.warning(f"Optim warning: {optim_result.message}", icon="⚠️")

                            add_log_message("info", "-" * 50 + "\nBest result:");
                            p_optimal = optim_result.x; final_objective_value = optim_result.fun; final_mse_display = final_objective_value if np.isfinite(final_objective_value) and final_objective_value < HUGE_PENALTY else np.nan
                            add_log_message("info", f"  Optimal MSE (Objective): {final_mse_display:.4e}")
                            optimal_thickness_nm = p_optimal[0]; add_log_message("info", f"  Optimal Thickness: {optimal_thickness_nm:.3f} nm");
                            idx_start = 1; n_values_opt_final = p_optimal[idx_start : idx_start + num_knots_n]; idx_start += num_knots_n; log_k_values_opt_final = p_optimal[idx_start : idx_start + num_knots_k]

                            n_spline_final = CubicSpline(fixed_n_knot_lambdas, n_values_opt_final, bc_type='natural', extrapolate=True); log_k_spline_final = CubicSpline(fixed_k_knot_lambdas, log_k_values_opt_final, bc_type='natural', extrapolate=True)
                            n_final_array_recalc = np.full_like(current_target_lambda, np.nan); k_final_array_recalc = np.full_like(current_target_lambda, np.nan); T_stack_final_calc = np.full_like(current_target_lambda, np.nan); T_norm_final_calc = np.full_like(current_target_lambda, np.nan)
                            valid_lambda_mask_for_calc = (current_target_lambda >= substrate_min_limit) & np.isfinite(current_target_lambda) & (current_target_lambda > 0)
                            if np.any(valid_lambda_mask_for_calc):
                                lambda_to_eval = current_target_lambda[valid_lambda_mask_for_calc]
                                n_final_array_recalc[valid_lambda_mask_for_calc] = n_spline_final(lambda_to_eval);
                                k_final_array_recalc[valid_lambda_mask_for_calc] = np.exp(log_k_spline_final(lambda_to_eval))
                                n_final_array_recalc = np.clip(n_final_array_recalc, 1.0, N_KNOT_VALUE_BOUNDS[1]);
                                k_final_array_recalc = np.clip(k_final_array_recalc, 0.0, math.exp(LOG_K_KNOT_VALUE_BOUNDS[1]))
                                n_final_array_recalc[~np.isfinite(n_final_array_recalc)] = np.nan; k_final_array_recalc[~np.isfinite(k_final_array_recalc)] = np.nan
                            valid_nk_final_mask = np.isfinite(n_final_array_recalc) & np.isfinite(k_final_array_recalc); valid_nsub_mask_recalc = np.isfinite(nSub_target_array_full);
                            valid_indices_for_T_calc = np.where(valid_nk_final_mask & valid_nsub_mask_recalc)[0]
                            for i in valid_indices_for_T_calc:
                                l_val = current_target_lambda[i]; nMono_val = n_final_array_recalc[i] - 1j * k_final_array_recalc[i]; nSub_val = nSub_target_array_full[i]
                                try:
                                    _, Ts_stack_calc, _ = calculate_monolayer_lambda(l_val, nMono_val, optimal_thickness_nm, nSub_val); _, Ts_sub_calc, _ = calculate_monolayer_lambda(l_val, 1.0 + 0j, 0.0, nSub_val)
                                    if np.isfinite(Ts_stack_calc): T_stack_final_calc[i] = np.clip(Ts_stack_calc, 0.0, 1.0)
                                    T_norm_calc = np.nan
                                    if np.isfinite(Ts_sub_calc):
                                        if Ts_sub_calc > SMALL_EPSILON: T_norm_calc = Ts_stack_calc / Ts_sub_calc
                                        elif abs(Ts_stack_calc) < SMALL_EPSILON : T_norm_calc = 0.0
                                    if np.isfinite(T_norm_calc): T_norm_final_calc[i] = np.clip(T_norm_calc, 0.0, 2.0)
                                except Exception: pass

                            calc_value_for_mse = T_norm_final_calc if current_target_type == 'T_norm' else T_stack_final_calc;
                            combined_valid_mask_for_mse = mask_used_in_optimization & np.isfinite(calc_value_for_mse)
                            recalc_mse_final = np.nan; percent_good_fit = np.nan; quality_label = "N/A"; mse_pts_count = np.sum(combined_valid_mask_for_mse)
                            if mse_pts_count > 0 :
                                recalc_mse_final = np.mean((calc_value_for_mse[combined_valid_mask_for_mse] - current_target_value[combined_valid_mask_for_mse])**2);
                                abs_delta = np.abs(calc_value_for_mse[combined_valid_mask_for_mse] - current_target_value[combined_valid_mask_for_mse]);
                                delta_threshold = 0.0025
                                points_below_threshold = np.sum(abs_delta < delta_threshold);
                                percent_good_fit = (points_below_threshold / mse_pts_count) * 100.0

                                if percent_good_fit >= 90:
                                    quality_label = "Excellent"
                                elif percent_good_fit >= 70:
                                    quality_label = "Good"
                                elif percent_good_fit >= 50:
                                    quality_label = "Fair"
                                else:
                                    quality_label = "Poor"

                                add_log_message("info", f"  Final MSE ({current_target_type}, {mse_pts_count} pts in range): {recalc_mse_final:.4e}\n" + "-"*20 + " Fit Quality " + "-"*20 + f"\n  Range [{effective_lambda_min:.1f}-{effective_lambda_max:.1f}] nm, {mse_pts_count} valid pts\n  Points |delta|<{delta_threshold*100:.2f}%: {percent_good_fit:.1f}% ({points_below_threshold}/{mse_pts_count})\n  -> Rating: {quality_label}")
                            else:
                                add_log_message("warning", f"Cannot recalculate Final MSE/Quality for range [{effective_lambda_min:.1f}-{effective_lambda_max:.1f}] nm.")
                            add_log_message("info", "-" * 50)

                            plot_lambda_array_final = np.linspace(st.session_state.lambda_min_file, st.session_state.lambda_max_file, 500)
                            final_results_dict = {
                                'final_spectra': { 'l': current_target_lambda, 'T_stack_calc': T_stack_final_calc, 'T_norm_calc': T_norm_final_calc, 'MSE_Optimized': final_mse_display, 'MSE_Recalculated': recalc_mse_final, 'percent_good_fit': percent_good_fit, 'quality_label': quality_label },
                                'best_params': { 'thickness_nm': optimal_thickness_nm, 'num_knots_n': num_knots_n, 'num_knots_k': num_knots_k, 'n_knot_values': n_values_opt_final, 'log_k_knot_values': log_k_values_opt_final, 'n_knot_lambdas': fixed_n_knot_lambdas, 'k_knot_lambdas': fixed_k_knot_lambdas, 'knot_distribution': "1/λ²" if use_inv_lambda_sq_distrib else "1/λ", 'substrate_name': selected_substrate, 'effective_lambda_min': effective_lambda_min, 'effective_lambda_max': effective_lambda_max },
                                'plot_lambda_array': plot_lambda_array_final, 'model_str_base': "Spline Fit",
                                'result_data_table': {
                                    'lambda (nm)': current_target_lambda,
                                    f'n (Fit)': n_final_array_recalc,
                                    f'k (Fit)': k_final_array_recalc,
                                    'Thickness (nm)': np.full_like(current_target_lambda, optimal_thickness_nm),
                                    f'n Sub ({selected_substrate})': nSub_target_array_full,
                                    f'Target {current_target_type} (%) (Used)': np.where(mask_used_in_optimization, current_target_value * 100.0, np.nan),
                                    f'Target {current_target_type} (%) (Full)': current_target_value * 100.0,
                                    'Calc T (%)': T_stack_final_calc * 100.0,
                                    'Calc T Norm (%)': T_norm_final_calc * 100.0,
                                    'Delta T (%)': (T_stack_final_calc - current_target_value)*100.0 if current_target_type == 'T' else np.full_like(current_target_lambda, np.nan),
                                    'Delta T Norm (%)': (T_norm_final_calc - current_target_value)*100.0 if current_target_type == 'T_norm' else np.full_like(current_target_lambda, np.nan),
                                }
                            }
                            st.session_state.optim_results = final_results_dict

                            st.session_state.fig_compare_results = plot_spectra_vs_target(
                                res=final_results_dict['final_spectra'], target=st.session_state.target_data, best_params_info=final_results_dict['best_params'],
                                model_str_base=final_results_dict['model_str_base'], effective_lambda_min=final_results_dict['best_params']['effective_lambda_min'], effective_lambda_max=final_results_dict['best_params']['effective_lambda_max']
                            )
                            st.session_state.fig_nk_results = plot_nk_final(final_results_dict['best_params'], final_results_dict['plot_lambda_array'])
                            st.rerun()

                        except Exception as e_optim: st.error(f"Optimization Error: {e_optim}", icon="🚨"); add_log_message("error", f"ERROR during optimization: {e_optim}"); traceback.print_exc()

        if st.session_state.optim_results:
            results = st.session_state.optim_results
            st.divider();
            st.subheader("Optimization Metrics")
            col_res1a, col_res2a, col_res3a = st.columns(3)
            with col_res1a: st.metric("Optimal Thickness", f"{results['best_params']['thickness_nm']:.3f} nm")
            with col_res2a:
                mse_disp = results['final_spectra']['MSE_Recalculated'];
                st.metric("Final MSE (in range)", f"{mse_disp:.4e}" if np.isfinite(mse_disp) else "N/A")
            with col_res3a:
                quality_label = results['final_spectra']['quality_label']
                percent_good_fit = results['final_spectra']['percent_good_fit']
                if np.isfinite(percent_good_fit): st.metric("Fit Quality Rating", f"{quality_label}", help=f"Based on {percent_good_fit:.1f}% points within optim. range having |Calc - Target| < 0.25%")
                else: st.metric("Fit Quality Rating", "N/A")

            st.subheader("Result Plots")
            if st.session_state.fig_compare_results: st.pyplot(st.session_state.fig_compare_results)
            else: st.warning("Comparison plot not available.")
            if st.session_state.fig_nk_results: st.pyplot(st.session_state.fig_nk_results)
            else: st.warning("N/K plot not available.")

            st.subheader("Result Data")
            with st.expander("Show Result Data Table (Full Range)"):
                if 'result_data_table' in results:
                    try:
                        df_display = pd.DataFrame(results['result_data_table'])
                        st.dataframe(df_display, use_container_width=True)
                    except Exception as e_df:
                        st.warning(f"Could not display result table: {e_df}")
                        add_log_message("error", f"Error displaying DataFrame: {e_df}")
                else:
                    st.warning("Result data table not available.")
        elif run_button and not valid_params:
             st.warning("Optimization could not run due to parameter errors. Please check configuration and logs.", icon="⚠️")
        elif not run_button and not st.session_state.optim_results:
             st.info("Configure settings, load data, and click 'Run Optimization' to see results here.")

    st.divider()
    col_foot1, col_foot2 = st.columns(2)
    with col_foot1:
        with st.expander("Show User Access Log", expanded=False):
            display_user_log()
        display_log()

    with col_foot2:
        help_text_en = """
        **User Manual**

        **Goal:** Determine optical properties (n, k) and thickness (d) of a monolayer on a known substrate by fitting calculated transmission to experimental target data.

        **Tabs:**
        1.  **Configuration:** Set Substrate, Optimization Lambda Range, Thickness Range, and Advanced Parameters.
        2.  **Target Data:** Select Target Type (T_norm/T_sample) and Upload a **.csv** file (λ (nm), Target Value). Schema and Target Plot are shown.
        3.  **Run & Results:** Click "▶ Run Optimization". View Metrics, Result Plots (Comparison, n/k), and the Result Data Table.

        **Tips:** Check logs (below) for details/errors. Ensure λ range is valid. Use caching for plots where possible. Set `workers=1` in Advanced Settings if pickling errors occur.
        """
        with st.expander("Help / Instructions", expanded=False):
            st.markdown(help_text_en)

        user_name = st.session_state.user_name
        user_email = st.session_state.user_email
        if user_name and user_email: user_display_name = f"{user_name} ({user_email})"
        elif user_name: user_display_name = user_name
        elif user_email: user_display_name = user_email
        else: user_display_name = "Guest"
        st.caption(f"Monolayer Optimizer v1.4.3-tabs-opt - Welcome, {user_display_name}!")
