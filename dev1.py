# -*- coding: utf-8 -*-
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
# Removed smtplib and email imports
# Added csv and datetime imports for logging
import csv
from datetime import datetime


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the log file name
USER_LOG_FILE = 'user_log.csv'

# --- Constants and Setup ---
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

# --- Utility and Calculation Functions ---
def get_substrate_min_lambda(substrate_name):
    return SUBSTRATE_MIN_LAMBDA.get(substrate_name, 200.0)

# MODIFIED: cache=True
@numba.jit(nopython=True, cache=True)
def sellmeier_calc(l_um_sq, B1, C1, B2, C2, B3, C3):
    n_sq_minus_1 = 0.0
    # Only add terms if C is non-zero and away from l_um_sq to avoid division by zero
    if abs(C1) > SMALL_EPSILON and abs(l_um_sq - C1) > SMALL_EPSILON:
         n_sq_minus_1 += (B1 * l_um_sq / (l_um_sq - C1))
    if abs(C2) > SMALL_EPSILON and abs(l_um_sq - C2) > SMALL_EPSILON:
         n_sq_minus_1 += (B2 * l_um_sq / (l_um_sq - C2))
    if abs(C3) > SMALL_EPSILON and abs(l_um_sq - C3) > SMALL_EPSILON:
         n_sq_minus_1 += (B3 * l_um_sq / (l_um_sq - C3))
    # Handle cases like B=0 or C=0 gracefully
    n_sq = n_sq_minus_1 + 1.0
    # Ensure physical plausibility (n^2 >= 0)
    if n_sq < 0: n_sq = 0 # Or handle as error/NaN? Setting to 0 yields n=0
    return np.sqrt(n_sq)


# MODIFIED: cache=True
@numba.jit(nopython=True, cache=True)
def get_n_substrate(substrate_id, wavelength_nm):
    l_um = wavelength_nm / 1000.0
    l_um_sq = l_um * l_um
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
        # --- NOUVEAUX COEFFICIENTS SAPPHIRE ---
        B1 = 2.003059; C1 = 0.011694
        B2 = 0.360392; C2 = 1000.0  # Note: C2 large means this term approaches B2*l_um_sq / l_um_sq = B2
        B3 = 0.0;      C3 = 1.0     # B3=0 means this term is zero
        # -------------------------------------
        return sellmeier_calc(l_um_sq, B1, C1, B2, C2, B3, C3)
    elif substrate_id == 4: # B270i
        min_wl = 400.0
        if wavelength_nm < min_wl: return np.nan
        B1=0.90110328; C1=0.0045578115; B2=0.39734436; C2=0.016601149; B3=0.94615601; C3=111.88593
        return sellmeier_calc(l_um_sq, B1, C1, B2, C2, B3, C3)
    else: # Default to SiO2 if unknown
        min_wl = 230.0
        if wavelength_nm < min_wl: return np.nan
        B1=0.6961663; C1=0.0684043**2; B2=0.4079426; C2=0.1162414**2; B3=0.8974794; C3=9.896161**2
        return sellmeier_calc(l_um_sq, B1, C1, B2, C2, B3, C3)

# MODIFIED: cache=True
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

# MODIFIED: cache=True
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

        # Apply bounds strictly INSIDE the loop, per point
        n_calc = max(1.0, n_calc) # Ensure n >= 1 physically
        k_calc = max(0.0, k_calc) # Ensure k >= 0 physically

        # Penalize if OUTSIDE the defined knot value bounds (soft penalty in objective, hard here?)
        # Using HUGE_PENALTY ensures these solutions are discarded by the optimizer.
        if not (n_min_bound <= n_calc <= n_max_bound): return HUGE_PENALTY
        if not (k_min_bound <= k_calc <= k_max_bound): return HUGE_PENALTY
        # Also check for NaN/Inf that might slip through spline extrapolation
        if not (np.isfinite(n_calc) and np.isfinite(k_calc)): return HUGE_PENALTY

        nMono_complex_val = n_calc - 1j * k_calc

        # Calculate stack transmission
        _, Ts_stack, _ = calculate_monolayer_lambda(l_val, nMono_complex_val, current_thickness_nm, nSub_val)
        # If calculation fails for this point, we can't evaluate error for it
        if not np.isfinite(Ts_stack): continue # Skip this point or return penalty? Skip for now.

        calculated_value = np.nan
        if target_type_flag == 0: # T_norm
            # Calculate substrate-only transmission for normalization
            _, Ts_sub, _ = calculate_monolayer_lambda(l_val, 1.0 + 0j, 0.0, nSub_val)
            # If substrate transmission calculation fails, we can't normalize
            if not np.isfinite(Ts_sub): continue # Skip this point

            # Calculate normalized transmission
            if Ts_sub > SMALL_EPSILON:
                T_norm_calc = Ts_stack / Ts_sub
            else:
                # Avoid division by zero. If stack T is also near zero, T_norm is 0. Otherwise, it's a problem (infinite absorption?). Penalize?
                T_norm_calc = 0.0 if abs(Ts_stack) < SMALL_EPSILON else HUGE_PENALTY # Or maybe np.nan and let it be skipped? Use penalty.

            if not np.isfinite(T_norm_calc): return HUGE_PENALTY # Abort if T_norm is bad
            calculated_value = max(0.0, min(2.0, T_norm_calc)) # Clip T_norm (can exceed 1 slightly due to AR effects)

        elif target_type_flag == 1: # T_sample
            calculated_value = max(0.0, min(1.0, Ts_stack)) # Clip T_sample between 0 and 1

        else: # Unknown target type
            return HUGE_PENALTY # Should not happen if flag is 0 or 1

        # Calculate error if target is valid
        if np.isfinite(target_value_array[i]) and np.isfinite(calculated_value):
            error_i = (calculated_value - target_value_array[i])**2
            total_sq_error += error_i * weights_array[i] # Apply weight
            points_calculated += weights_array[i] # Accumulate total weight
        elif not np.isfinite(target_value_array[i]):
             # If target is NaN, just ignore this point (no error contribution)
            pass
        else:
             # If calc failed but target was valid -> HUGE PENALTY
             return HUGE_PENALTY


    # Average the weighted squared error over the points actually calculated
    if points_calculated <= SMALL_EPSILON:
        # No valid points were calculated or weighted sum is zero -> return penalty
        return HUGE_PENALTY
    return total_sq_error / points_calculated

# NOTE: Objective function itself is NOT jitted with Numba
def objective_func_spline_fixed_knots(p, num_knots_n, num_knots_k, l_array, nSub_array, target_value_array, weights_array, target_type_flag, fixed_n_knot_lambdas, fixed_k_knot_lambdas):
    """Objective function using cubic splines with fixed knot wavelengths."""
    expected_len = 1 + num_knots_n + num_knots_k
    if len(p) != expected_len:
        #print(f"DEBUG: Param length mismatch. Expected {expected_len}, got {len(p)}")
        return HUGE_PENALTY # Parameter vector length mismatch

    current_thickness_nm = p[0]
    idx_start = 1
    n_knot_values = p[idx_start : idx_start + num_knots_n]; idx_start += num_knots_n
    log_k_knot_values = p[idx_start : idx_start + num_knots_k]

    # Basic sanity checks on parameter values
    if current_thickness_nm < 0: return HUGE_PENALTY # Thickness must be non-negative
    # Check against defined bounds for knot values
    if np.any(n_knot_values < N_KNOT_VALUE_BOUNDS[0] - SMALL_EPSILON) or \
       np.any(n_knot_values > N_KNOT_VALUE_BOUNDS[1] + SMALL_EPSILON) or \
       np.any(log_k_knot_values < LOG_K_KNOT_VALUE_BOUNDS[0] - SMALL_EPSILON) or \
       np.any(log_k_knot_values > LOG_K_KNOT_VALUE_BOUNDS[1] + SMALL_EPSILON):
        #print("DEBUG: Knot value out of bounds.")
        return HUGE_PENALTY

    try:
        # Need at least 2 knots for CubicSpline (scipy requirement)
        if num_knots_n < 2 or num_knots_k < 2 : return HUGE_PENALTY

        # Create spline interpolators for n and log(k)
        n_spline = CubicSpline(fixed_n_knot_lambdas, n_knot_values, bc_type='natural', extrapolate=True)
        log_k_spline = CubicSpline(fixed_k_knot_lambdas, log_k_knot_values, bc_type='natural', extrapolate=True)

        # Calculate n and k arrays for the target wavelengths using the splines
        n_calc_array = n_spline(l_array)
        k_calc_array = np.exp(log_k_spline(l_array)) # k = exp(log(k))

        # Get effective min/max bounds for n and k based on knot bounds for the Numba function
        # Note: Splines might extrapolate outside these, but Numba function will clip/penalize
        n_min_req = N_KNOT_VALUE_BOUNDS[0]
        n_max_req = N_KNOT_VALUE_BOUNDS[1]
        k_min_req = math.exp(LOG_K_KNOT_VALUE_BOUNDS[0])
        k_max_req = math.exp(LOG_K_KNOT_VALUE_BOUNDS[1])

        # Call the Numba-optimized function to calculate the total MSE
        mse = calculate_total_error_numba(l_array, nSub_array, target_value_array,
                                          weights_array, target_type_flag,
                                          current_thickness_nm,
                                          n_calc_array, k_calc_array,
                                          n_min_req, n_max_req, k_min_req, k_max_req)
        #if mse > 1e20: print(f"DEBUG: High MSE {mse}")
        return mse

    except ValueError as e:
        # Catch potential errors during spline creation or evaluation (e.g., if knot lambdas aren't sorted)
        #print(f"DEBUG: ValueError in objective function: {e}")
        return HUGE_PENALTY
    except Exception as e:
        # Catch any other unexpected errors
        #print(f"DEBUG: Exception in objective function: {e}")
        # traceback.print_exc() # Uncomment for detailed debugging if needed
        return HUGE_PENALTY


def add_log_message(message_type, message):
    if 'log_messages' not in st.session_state: st.session_state.log_messages = []
    st.session_state.log_messages.append((message_type, message))
    if message_type == "info": logger.info(message)
    elif message_type == "warning": logger.warning(message)
    elif message_type == "error": logger.error(message)

def display_log():
    if 'log_messages' in st.session_state and st.session_state.log_messages:
        with st.expander("Log Messages", expanded=False):
            log_container = st.container()
            # Display logs in reverse chronological order (newest first)
            for msg_type, msg in reversed(st.session_state.log_messages):
                if msg_type == "info": log_container.info(msg, icon="ℹ️")
                elif msg_type == "warning": log_container.warning(msg, icon="⚠️")
                elif msg_type == "error": log_container.error(msg, icon="🚨")
                else: log_container.text(msg) # Fallback for unknown types

def reset_log():
    st.session_state.log_messages = []

def plot_target_only(target_data_to_plot, target_filename_base):
    if target_data_to_plot is None or 'lambda' not in target_data_to_plot or len(target_data_to_plot['lambda']) == 0:
        add_log_message("warning", "No target data available to plot.")
        return None

    target_type = target_data_to_plot.get('target_type', 'T_norm') # Default if missing
    target_values = target_data_to_plot.get('target_value', None)
    target_l = target_data_to_plot['lambda']

    if target_values is None:
        add_log_message("warning", "Target values are missing in the data.")
        return None

    fig_target, ax_target = plt.subplots(1, 1, figsize=(8, 6));
    short_filename = target_filename_base if target_filename_base else "Target"
    if target_type == 'T_norm':
        plot_label = 'Target T Norm (%)'; y_label = 'Normalized Transmission (%)'; title_suffix = "T Norm (%)"; y_lim_top = 110
    else: # Assumed T_sample
        plot_label = 'Target T (%)'; y_label = 'Transmission (%)'; title_suffix = "T Sample (%)"; y_lim_top = 105

    ax_target.set_title(f"Target Data ({short_filename}) - {title_suffix}")
    ax_target.set_xlabel('λ (nm)');
    ax_target.grid(True, which='both', linestyle=':', linewidth=0.5); ax_target.minorticks_on();

    # Ensure only finite values are plotted
    valid_mask = np.isfinite(target_values) & np.isfinite(target_l)
    if np.any(valid_mask):
        ax_target.plot(target_l[valid_mask], target_values[valid_mask] * 100.0, '.', markersize=5, color='red', linestyle='none', label=plot_label)
        ax_target.set_ylabel(y_label)
        # Adjust y-limits based on plotted data
        min_y_data = np.min(target_values[valid_mask] * 100.0)
        max_y_data = np.max(target_values[valid_mask] * 100.0)
        y_padding = max(5, (max_y_data - min_y_data) * 0.05) if max_y_data > min_y_data else 5
        ax_target.set_ylim(bottom=max(-5, min_y_data - y_padding), top=min(y_lim_top + 10 , max_y_data + y_padding))
    else:
        add_log_message("warning", "No valid (finite) target data points to plot.")
        ax_target.set_ylabel('No Valid Data')
        ax_target.set_ylim(bottom=-5, top=y_lim_top) # Default limits if no data

    # Use lambda range from file if available, otherwise from valid plotted data
    lambda_min_plot = np.nanmin(target_l[valid_mask]) if 'lambda_min_file' not in st.session_state and np.any(valid_mask) else st.session_state.get('lambda_min_file', 0)
    lambda_max_plot = np.nanmax(target_l[valid_mask]) if 'lambda_max_file' not in st.session_state and np.any(valid_mask) else st.session_state.get('lambda_max_file', 1000)
    if lambda_min_plot < lambda_max_plot: ax_target.set_xlim(lambda_min_plot, lambda_max_plot)

    ax_target.legend(fontsize='small')
    plt.tight_layout()
    return fig_target

def plot_nk_final(best_params_info, plot_lambda_array):
    fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
    ax2 = ax1.twinx() # Share the same x-axis

    # Extract data from the results dictionary
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
        # Recreate splines from optimal knot values
        n_spline = CubicSpline(fixed_n_lambdas, n_values_opt_final, bc_type='natural', extrapolate=True)
        log_k_spline = CubicSpline(fixed_k_lambdas, log_k_values_opt_final, bc_type='natural', extrapolate=True)
        # Evaluate splines over the plotting range
        n_plot = n_spline(plot_lambda_array); k_plot = np.exp(log_k_spline(plot_lambda_array))

        color1='tab:red'; color2='tab:blue'

        # Plot k on ax1 (left y-axis)
        ax1.set_xlabel('λ (nm)'); ax1.set_ylabel("k Coeff.", color=color1);
        ax1.plot(plot_lambda_array, k_plot, color=color1, linestyle='--', linewidth=1.5, label='k (Final)')
        # Plot k knots as markers
        ax1.plot(fixed_k_lambdas, np.exp(log_k_values_opt_final), 's', color=color1, markersize=6, fillstyle='none', label=f'k Knots ({num_k})')
        ax1.tick_params(axis='y', labelcolor=color1); ax1.grid(True,which='major',ls=':',lw=0.5,axis='y',color=color1)

        # Plot n on ax2 (right y-axis)
        ax2.set_ylabel('n Index', color=color2)
        ax2.plot(plot_lambda_array, n_plot, color=color2, linestyle='-', linewidth=1.5, label='n (Final)')
        # Plot n knots as markers
        ax2.plot(fixed_n_lambdas, n_values_opt_final, 'o', color=color2, markersize=6, fillstyle='none', label=f'n Knots ({num_n})')
        ax2.tick_params(axis='y', labelcolor=color2) # No grid for ax2 to avoid clutter

        # Common x-axis grid
        ax1.grid(True,which='both',ls=':',lw=0.5,axis='x');

        # Dynamic Y-axis scaling for k
        k_min_req = math.exp(LOG_K_KNOT_VALUE_BOUNDS[0]); k_max_req = math.exp(LOG_K_KNOT_VALUE_BOUNDS[1])
        min_k_plot_val = np.nanmin(k_plot) if k_plot is not None and np.any(np.isfinite(k_plot)) else k_min_req
        max_k_plot_val = np.nanmax(k_plot) if k_plot is not None and np.any(np.isfinite(k_plot)) else k_max_req
        min_k_disp = max(SMALL_EPSILON * 0.1, k_min_req * 0.5) # Avoid exactly zero for log scale
        max_k_disp = max(k_max_req * 2.0, max_k_plot_val * 1.2) if np.isfinite(max_k_plot_val) else k_max_req * 2.0

        # Use log scale for k if range is large
        use_log_k = (max_k_disp / min_k_disp) > 100 if min_k_disp > 0 and max_k_disp > 0 else False
        if use_log_k:
            ax1.set_yscale('log'); ax1.set_ylim(bottom=min_k_disp, top=max_k_disp); ax1.set_ylabel("k Coeff. [Log]", color=color1)
        else:
            ax1.set_yscale('linear'); k_range = k_max_req - k_min_req
            k_low_lim = max(0, k_min_req - 0.1 * k_range) if k_range > 0 else 0
            k_high_lim = k_max_req + 0.1 * k_range if k_range > 0 else k_max_req * 1.1
            # Ensure plotted values fit, even if outside knot bounds (spline extrapolation)
            ax1.set_ylim(bottom=k_low_lim, top=max(k_high_lim, max_k_plot_val * 1.1 if np.isfinite(max_k_plot_val) else k_high_lim)); ax1.set_ylabel("k Coeff.", color=color1)


        # Dynamic Y-axis scaling for n
        n_min_req = N_KNOT_VALUE_BOUNDS[0]; n_max_req = N_KNOT_VALUE_BOUNDS[1]
        n_range = n_max_req - n_min_req
        n_low_lim = n_min_req - 0.1 * n_range if n_range > 0 else n_min_req * 0.95
        n_high_lim = n_max_req + 0.1 * n_range if n_range > 0 else n_max_req * 1.05
        min_n_plot_val = np.nanmin(n_plot) if n_plot is not None and np.any(np.isfinite(n_plot)) else n_min_req
        max_n_plot_val = np.nanmax(n_plot) if n_plot is not None and np.any(np.isfinite(n_plot)) else n_max_req
        # Ensure plotted values fit, even if outside knot bounds
        min_n_ylim = min(n_low_lim, min_n_plot_val * 0.98 if np.isfinite(min_n_plot_val) else n_low_lim)
        max_n_ylim = max(n_high_lim, max_n_plot_val * 1.02 if np.isfinite(max_n_plot_val) else n_high_lim)
        ax2.set_ylim(bottom=min_n_ylim, top=max_n_ylim)


        # Set X limits based on effective optimization range
        if lambda_min_eff is not None and lambda_max_eff is not None and lambda_min_eff < lambda_max_eff :
            ax1.set_xlim(lambda_min_eff, lambda_max_eff)

        ax1.set_title('Final Optimized n/k Indices')
        # Combine legends from both axes
        handles1, labels1 = ax1.get_legend_handles_labels(); handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc='best', fontsize='small')
        plt.tight_layout() # Adjust layout to prevent overlap
        return fig

    except Exception as e:
        add_log_message("error", f"Failed to generate final n/k plot: {e}")
        # traceback.print_exc() # Uncomment for debugging
        return None


def plot_spectra_vs_target(res, target=None, best_params_info=None, model_str_base="Spline Fit", effective_lambda_min=None, effective_lambda_max=None):
    try:
        fig, ax = plt.subplots(1, 1, figsize=(9, 7)); ax_delta = ax.twinx() # Delta axis

        # --- Extract details for labels/title ---
        num_knots_n = best_params_info.get('num_knots_n', '?'); num_knots_k = best_params_info.get('num_knots_k', '?'); knot_distrib = best_params_info.get('knot_distribution', '?')
        model_str = f"{model_str_base} ({num_knots_n}n/{num_knots_k}k Knots, {knot_distrib})"
        target_type = target.get('target_type', 'T_norm') if target else 'T_norm'

        # Determine plot labels based on target type
        if target_type == 'T_norm':
            comparison_label = 'Normalized T'; target_label_suffix = 'T Norm (%)'; calc_label_suffix = 'T Norm (%)'; y_label = 'Normalized Transmission (%)'; target_value_key = 'target_value'; calc_value_key = 'T_norm_calc'; y_lim_top = 110
        else: # Assumed T_sample
            comparison_label = 'Sample T'; target_label_suffix = 'T (%)'; calc_label_suffix = 'T (%)'; y_label = 'Transmission (%)'; target_value_key = 'target_value'; calc_value_key = 'T_stack_calc'; y_lim_top = 105

        # --- Build Title ---
        title_base=f'{comparison_label} Comparison ({model_str})'; params_list = []
        if best_params_info and 'thickness_nm' in best_params_info: params_list.append(f"d={best_params_info['thickness_nm']:.2f}nm")
        if best_params_info and 'substrate_name' in best_params_info: params_list.append(f"Sub={best_params_info['substrate_name']}")
        if effective_lambda_min is not None and effective_lambda_max is not None: params_list.append(f"λ=[{effective_lambda_min:.0f}-{effective_lambda_max:.0f}]nm")
        if params_list: title_base += f' - {", ".join(params_list)}'
        title = title_base;
        # Add MSE info if available
        if 'MSE_Recalculated' in res and res['MSE_Recalculated'] is not None and np.isfinite(res['MSE_Recalculated']): title += f"\nFinal MSE (in Range): {res['MSE_Recalculated']:.3e}"
        elif 'MSE_Optimized' in res and res['MSE_Optimized'] is not None and np.isfinite(res['MSE_Optimized']): title += f"\nOptim. Objective MSE: {res['MSE_Optimized']:.3e}" # Fallback

        ax.set_title(title, fontsize=10); # Smaller font for potentially long title
        ax.set_xlabel('λ (nm)'); ax.set_ylabel(y_label); ax.grid(True, which='both', linestyle=':', linewidth=0.5); ax.minorticks_on(); ax.set_ylim(bottom=-5, top=y_lim_top)

        # --- Plot Calculated Spectrum ---
        line_calc = None; calc_l = res.get('l'); calc_y = res.get(calc_value_key)
        if calc_l is not None and calc_y is not None:
            valid_calc_mask = np.isfinite(calc_y) & np.isfinite(calc_l)
            # Plot only within the original file's full lambda range for context
            # plot_mask_calc = valid_calc_mask & (calc_l >= st.session_state.get('lambda_min_file',0)) & (calc_l <= st.session_state.get('lambda_max_file', np.inf))
            plot_mask_calc = valid_calc_mask # Plot all valid calculated points
            if np.any(plot_mask_calc):
                line_calc, = ax.plot(calc_l[plot_mask_calc], calc_y[plot_mask_calc] * 100.0, label=f'Calc {calc_label_suffix}', linestyle='-', color='darkblue', linewidth=1.5);

        # --- Plot Target Data ---
        line_tgt = None; target_l_valid, target_y_valid = None, None
        if target is not None and 'lambda' in target and target_value_key in target and len(target['lambda']) > 0:
            target_l_valid_orig = target['lambda']; target_y_valid_fraction = target[target_value_key]
            # Plot only valid target points within the original file's full lambda range
            valid_target_mask = np.isfinite(target_y_valid_fraction) & np.isfinite(target_l_valid_orig)
            #final_target_mask = valid_target_mask & (target_l_valid_orig >= st.session_state.get('lambda_min_file',0)) & (target_l_valid_orig <= st.session_state.get('lambda_max_file', np.inf))
            final_target_mask = valid_target_mask # Plot all valid target points
            if np.any(final_target_mask):
                target_l_valid = target_l_valid_orig[final_target_mask]; target_y_valid = target_y_valid_fraction[final_target_mask] * 100.0
                line_tgt, = ax.plot(target_l_valid, target_y_valid, 'o', markersize=4, color='red', fillstyle='none', label=f'Target {target_label_suffix}');

        # --- Plot Delta (Calc - Target) on ax_delta ---
        line_delta = None; delta_t_perc = np.full_like(calc_l, np.nan) if calc_l is not None else np.array([])
        # Ensure we have necessary data for delta calculation
        if calc_l is not None and calc_y is not None and target_l_valid is not None and target_y_valid is not None and len(target_l_valid) > 1:
            calc_y_perc = calc_y * 100.0
            # Interpolate target data onto the calculated lambda points for direct comparison
            try:
                # Use linear interpolation, don't extrapolate outside target range
                 target_y_perc_interp = np.interp(calc_l, target_l_valid, target_y_valid, left=np.nan, right=np.nan)
                 delta_t_perc_full = calc_y_perc - target_y_perc_interp

                 # Plot delta only within the OPTIMIZATION range where comparison is meaningful
                 valid_delta_mask = np.isfinite(delta_t_perc_full) & np.isfinite(calc_l)
                 plot_mask_delta = (calc_l >= effective_lambda_min) & (calc_l <= effective_lambda_max) & valid_delta_mask

                 if np.any(plot_mask_delta):
                     line_delta, = ax_delta.plot(calc_l[plot_mask_delta], delta_t_perc_full[plot_mask_delta], label='ΔT (%) [Calc - Target, Optim. Range]', linestyle=':', color='green', linewidth=1.2, zorder=-5); # Send to back
                     # Adjust delta y-axis limits dynamically
                     min_delta = np.min(delta_t_perc_full[plot_mask_delta]); max_delta = np.max(delta_t_perc_full[plot_mask_delta])
                     padding = max(1.0, abs(max_delta - min_delta) * 0.1) if max_delta != min_delta else 1.0 # Min padding 1%
                     ax_delta.set_ylim(min_delta - padding, max_delta + padding); ax_delta.set_ylabel('ΔT (%) [Optim. Range]', color='green'); ax_delta.tick_params(axis='y', labelcolor='green'); ax_delta.grid(True, axis='y', linestyle='-.', linewidth=0.5, color='lightgreen', alpha=0.6)
                 else:
                      # If no valid delta points in range, setup axis but maybe hide ticks?
                     ax_delta.set_ylabel('ΔT (%) [Optim. Range]', color='green'); ax_delta.tick_params(axis='y', labelcolor='green'); ax_delta.set_yticks([]) # Hide y-ticks if no data to plot

            except Exception as e_interp:
                 add_log_message("warning", f"Could not interpolate target for delta plot: {e_interp}")
                 ax_delta.set_ylabel('ΔT (%) [Optim. Range]', color='green'); ax_delta.tick_params(axis='y', labelcolor='green'); ax_delta.set_yticks([])


        # --- Set X Limits ---
        # Prioritize full file range, fallback to optimization range, fallback to data range
        file_lambda_min = st.session_state.get('lambda_min_file')
        file_lambda_max = st.session_state.get('lambda_max_file')
        if file_lambda_min is not None and file_lambda_max is not None and file_lambda_min < file_lambda_max:
             ax.set_xlim(file_lambda_min, file_lambda_max)
        elif effective_lambda_min is not None and effective_lambda_max is not None and effective_lambda_min < effective_lambda_max:
             ax.set_xlim(effective_lambda_min, effective_lambda_max) # Fallback to optim range
        else:
             # Fallback to min/max of calculated lambda if available
             min_l_plot_fallback = np.nanmin(res.get('l', [300])); max_l_plot_fallback = np.nanmax(res.get('l', [1000]));
             if np.isfinite(min_l_plot_fallback) and np.isfinite(max_l_plot_fallback) and min_l_plot_fallback < max_l_plot_fallback:
                 ax.set_xlim(min_l_plot_fallback, max_l_plot_fallback)


        # --- Combine Legends ---
        handles1, labels1 = ax.get_legend_handles_labels(); handles2, labels2 = ax_delta.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc='best', fontsize='small')

        # --- Add Quality Annotation ---
        percent_good_fit = res.get('percent_good_fit', np.nan); quality_label = res.get('quality_label', 'N/A')
        if np.isfinite(percent_good_fit):
            quality_text = f"Fit Quality (in Optim. Range): {quality_label}\n(<0.25% abs delta): {percent_good_fit:.1f}%"
            ax.text(0.98, 0.02, quality_text, transform=ax.transAxes, fontsize=10, ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', alpha=0.8))


        plt.tight_layout()
        return fig
    except Exception as e_plot:
        add_log_message("error", f"Failed to generate final spectra plot: {e_plot}")
        # traceback.print_exc() # For debugging
        return None


def plot_substrate_indices():
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    lambda_nm = np.linspace(200, 2000, 500) # Plotting range

    for idx, name in enumerate(SUBSTRATE_LIST):
        min_wl_sub = get_substrate_min_lambda(name)
        try:
            # Calculate n values using the Numba function
            n_values = np.array([get_n_substrate(idx, l) for l in lambda_nm])

            # Plot valid range with solid line
            valid_mask = lambda_nm >= min_wl_sub
            line, = ax.plot(lambda_nm[valid_mask], n_values[valid_mask], label=f"{name} (≥{min_wl_sub:.0f} nm)", linewidth=1.5)

            # Plot invalid range (below min_wl) with dashed/dotted line of same color
            invalid_mask = lambda_nm < min_wl_sub
            if np.any(invalid_mask):
                 # Recalculate n for the invalid part (might return NaN, which is fine)
                 n_invalid = np.array([get_n_substrate(idx, l) for l in lambda_nm[invalid_mask]])
                 # Plot only where n_invalid is finite (Numba func returns NaN below min_wl)
                 finite_invalid_mask = np.isfinite(n_invalid)
                 ax.plot(lambda_nm[invalid_mask][finite_invalid_mask], n_invalid[finite_invalid_mask], linestyle=':', color=line.get_color(), alpha=0.5)


        except Exception as e:
            add_log_message("warning", f"Could not calculate index for {name}. Error: {e}")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Refractive Index (n)")
    ax.set_title("Substrate Refractive Indices (Sellmeier - Dashed below nominal range)")
    ax.grid(True, which='major', linestyle='-', linewidth='0.5', color='gray')
    ax.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgray')
    ax.minorticks_on()
    ax.legend()
    ax.set_xlim(200, 2000) # Fixed x-axis range for comparison
    # You might want dynamic Y limits or set reasonable fixed ones
    # ax.set_ylim(1.4, 2.0) # Example fixed range
    plt.tight_layout()
    return fig

def draw_schema_matplotlib(target_type, substrate_name):
    fig, ax = plt.subplots(figsize=(5.5, 1.5)) # Adjusted size if needed
    fig.patch.set_alpha(0) # Transparent background for figure
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis('off') # No axes

    # Colors and fonts
    color_Mono = "#D6EAF8"; color_Sub = "#EAECEE"; outline_color = "#5D6D7E"
    layer_font_size = 7; medium_font_size = 7
    layer_font = {'size': layer_font_size, 'color': outline_color}
    medium_font = {'size': medium_font_size, 'weight': 'bold', 'color': outline_color}

    # --- Sample Side (Left) ---
    x_left = 10; stack_width = 25; mono_h = 8; sub_h = 8; y_sub_base = 5

    # Monolayer Rectangle + Text
    ax.add_patch(plt.Rectangle((x_left, y_sub_base + sub_h), stack_width, mono_h, facecolor=color_Mono, edgecolor=outline_color, linewidth=0.5))
    ax.text(x_left + stack_width/2, y_sub_base + sub_h + mono_h/2, "Monolayer", ha='center', va='center', fontdict=layer_font)
    # Substrate Rectangle + Text
    ax.add_patch(plt.Rectangle((x_left, y_sub_base), stack_width, sub_h, facecolor=color_Sub, edgecolor=outline_color, linewidth=0.5))
    ax.text(x_left + stack_width/2, y_sub_base + sub_h/2, f"Sub ({substrate_name})", ha='center', va='center', fontdict=layer_font)
    # Medium Text
    ax.text(x_left + stack_width/2, y_sub_base + sub_h + mono_h + 3, "Air (n≈1)", ha='center', va='top', fontdict=medium_font)
    ax.text(x_left + stack_width/2, y_sub_base - 3, "Air", ha='center', va='bottom', fontdict=medium_font)

    # Transmission Arrow + Label (T_sample or T_norm)
    arrow_x = x_left + stack_width/2; y_arrow_start = 28; y_arrow_end = 2
    ax.arrow(arrow_x, y_arrow_start, 0, y_arrow_end - y_arrow_start, head_width=3, head_length=2, fc='darkred', ec='darkred', length_includes_head=True, width=0.5)
    label_text = "T_sample" if target_type == 'T' else "T_norm"
    ax.text(arrow_x + 4, y_arrow_end + 5, label_text, ha='left', va='center', color='darkred', style='italic', size=8)


    # --- Reference Side (Right, only for T_norm) ---
    if target_type == 'T_norm':
        x_right = 100 - 10 - stack_width # Position for the right stack

        # Substrate Rectangle + Text (Reference)
        ax.add_patch(plt.Rectangle((x_right, y_sub_base), stack_width, sub_h, facecolor=color_Sub, edgecolor=outline_color, linewidth=0.5))
        ax.text(x_right + stack_width/2, y_sub_base + sub_h/2, f"Sub ({substrate_name})", ha='center', va='center', fontdict=layer_font)
        # Medium Text (Reference)
        ax.text(x_right + stack_width/2, y_sub_base + sub_h + 3, "Air (n≈1)", ha='center', va='top', fontdict=medium_font)
        ax.text(x_right + stack_width/2, y_sub_base - 3, "Air", ha='center', va='bottom', fontdict=medium_font)

        # Reference Transmission Arrow + Label (T_sub)
        arrow_x_right = x_right + stack_width / 2
        ax.arrow(arrow_x_right, y_arrow_start, 0, y_arrow_end - y_arrow_start, head_width=3, head_length=2, fc='darkred', ec='darkred', length_includes_head=True, width=0.5)
        ax.text(arrow_x_right + 4, y_arrow_end + 5, "T_sub", ha='left', va='center', color='darkred', style='italic', size=8)

        # Formula Text (Center)
        x_center = 50
        text_tnorm = f"{label_text} = T_sample / T_sub"
        ax.text(x_center, 15, text_tnorm, ha='center', va='center', style='italic', size=7, color=outline_color)


    return fig

def create_excel_file(results_data):
    """Creates an Excel file in memory from the results data."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_summary = results_data['summary']
        df_results = results_data['data']
        sheet_name_short = results_data.get('sheet_name', 'Results')[:30] # Ensure sheet name isn't too long

        # Write summary at the top
        df_summary.to_excel(writer, index=False, sheet_name=sheet_name_short, startrow=0, startcol=0)
        # Write detailed results below summary, leaving a blank row
        df_results.to_excel(writer, index=False, sheet_name=sheet_name_short, startrow=len(df_summary)+2, startcol=0)

        # Optional: Add formatting (e.g., column widths) here if needed using openpyxl workbook/worksheet objects
        # workbook = writer.book
        # worksheet = writer.sheets[sheet_name_short]
        # worksheet.column_dimensions['A'].width = 15 # Example

    output.seek(0) # Rewind the buffer
    return output.getvalue()


# --- Function to Log User Info to CSV ---
def log_user_access(timestamp, user_name, user_email):
    file_exists = os.path.isfile(USER_LOG_FILE)
    try:
        with open(USER_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header only if file is new or empty
            if not file_exists or os.path.getsize(USER_LOG_FILE) == 0:
                writer.writerow(['Timestamp', 'Name', 'Email'])
            writer.writerow([timestamp, user_name, user_email])
        return True
    except Exception as e:
        # Use Streamlit logging since this might happen before session state log is ready
        logger.error(f"Failed to write to user log file {USER_LOG_FILE}: {e}")
        # Optionally try to add to session state log if available
        if 'log_messages' in st.session_state:
            add_log_message("error", f"Failed to write to user log file {USER_LOG_FILE}: {e}")
        return False

# --- Function to Display User Log ---
def display_user_log():
    st.subheader("User Access Log")
    if os.path.isfile(USER_LOG_FILE):
        try:
            df_log = pd.read_csv(USER_LOG_FILE)
            st.metric("Total Access Records", len(df_log))
            # Display more recent logs first
            st.dataframe(df_log.iloc[::-1])
        except pd.errors.EmptyDataError:
             st.info("User log file is empty.")
        except Exception as e:
            st.error(f"Could not read user log file {USER_LOG_FILE}: {e}")
            add_log_message("error", f"Failed to read user log file: {e}")
    else:
        st.info("User log file not found or is empty.")


# --- Initialize Session State ---
if 'target_data' not in st.session_state: st.session_state.target_data = None
if 'target_filename_base' not in st.session_state: st.session_state.target_filename_base = None
if 'lambda_min_file' not in st.session_state: st.session_state.lambda_min_file = None
if 'lambda_max_file' not in st.session_state: st.session_state.lambda_max_file = None
if 'log_messages' not in st.session_state: st.session_state.log_messages = []
if 'optim_results' not in st.session_state: st.session_state.optim_results = None
if 'excel_bytes' in st.session_state: del st.session_state['excel_bytes'] # Removed
if 'excel_filename' in st.session_state: del st.session_state['excel_filename'] # Removed
if 'config_lambda_min' not in st.session_state: st.session_state.config_lambda_min = "---"
if 'config_lambda_max' not in st.session_state: st.session_state.config_lambda_max = "---"
if 'thickness_min' not in st.session_state: st.session_state.thickness_min = 300.0
if 'thickness_max' not in st.session_state: st.session_state.thickness_max = 600.0
if 'substrate_choice' not in st.session_state: st.session_state.substrate_choice = SUBSTRATE_LIST[0]
if 'target_type' not in st.session_state: st.session_state.target_type = "T_norm"
if 'last_loaded_source' not in st.session_state: st.session_state.last_loaded_source = None

if 'info_submitted' not in st.session_state:
    st.session_state.info_submitted = False
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""
if 'user_email' not in st.session_state:
    st.session_state.user_email = ""

default_advanced_params = {
    'num_knots_n': 6, 'num_knots_k': 6, 'use_inv_lambda_sq_distrib': False,
    'pop_size': 20, 'maxiter': 1500, 'tol': 0.001, 'atol': 0.0, # Relaxed default tolerances
    'mutation_min': 0.5, 'mutation_max': 1.2, 'recombination': 0.8, # Adjusted mutation range
    'strategy': 'best1bin', 'polish': True, 'updating': 'deferred', 'workers': -1 # Default to use all workers
}
if 'advanced_optim_params' not in st.session_state:
    st.session_state.advanced_optim_params = default_advanced_params.copy()


# --- Streamlit App Layout ---
st.set_page_config(page_title="Monolayer Optimizer", layout="wide")

# --- User Info Form ---
if not st.session_state.info_submitted:
    st.header("Welcome!")
    st.write("Please enter your name or your e-mail to continue.")
    st.info("Privacy Notice: Your details are logged for usage tracking when you access the application.", icon="ℹ️")

    with st.form("info_form"):
        name_input = st.text_input("Your name (optional)")
        email_input = st.text_input("Your email address") # Made mandatory for access

        submitted = st.form_submit_button("Access Application")

        if submitted:
            if email_input: # Check if email is provided
                st.session_state.user_name = name_input
                st.session_state.user_email = email_input
                st.session_state.info_submitted = True

                # --- Log user access to file ---
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_success = log_user_access(now_str, name_input, email_input)
                # Log message handled inside log_user_access or logger
                # --- End log access ---

                st.rerun() # Rerun to show main app
            else:
                st.error("Please provide at least an email address.")

# --- Main Application UI ---
else:
    user_display_name = st.session_state.user_name or st.session_state.user_email
    st.success(f"Welcome, {user_display_name}!")

    st.title("Monolayer Optical Properties Optimizer")

    # App Description
    with st.expander("📄 Application Description", expanded=False):
        st.markdown(r"""
        This Streamlit application determines the optical properties (refractive index $n(\lambda)$, extinction coefficient $k(\lambda)$) and the physical thickness ($d$) of a single thin layer (monolayer) deposited on a known substrate.

        The determination is achieved by fitting calculated optical transmission spectra (either normalized transmission $T_{norm}$ or direct sample transmission $T_{sample}$) to experimental target data provided by the user via a CSV file (or a default example file).
        The application uses cubic splines to model the dispersion of $n(\lambda)$ and $k(\lambda)$, and employs the differential evolution algorithm to find the optimal parameters (spline knot values and thickness $d$) that minimize the mean squared error between the calculated spectrum and the experimental target within a user-defined wavelength range, while respecting the physical limitations of the chosen substrate.

        The tool outputs the determined dispersions of $n(\lambda)$ and $k(\lambda)$, the optimal thickness $d$, comparison plots, and an assessment of the fit quality. User access is logged.

        ---
        *Developed by Fabien Lemarchand.*
        *For any questions, feedback, or potential collaborations, please contact:* [fabien.lemarchand@gmail.com](mailto:fabien.lemarchand@gmail.com)
        """)

    # Sidebar Configuration
    with st.sidebar:
        st.header("Configuration")
        prev_substrate = st.session_state.substrate_choice
        st.session_state.substrate_choice = st.selectbox(
            "Substrate Material:", SUBSTRATE_LIST,
            index=SUBSTRATE_LIST.index(st.session_state.substrate_choice) if st.session_state.substrate_choice in SUBSTRATE_LIST else 0
        )
        # Update lambda range hints if substrate changes and data is loaded
        if st.session_state.substrate_choice != prev_substrate and st.session_state.target_data is not None:
            substrate_min_wl = get_substrate_min_lambda(st.session_state.substrate_choice)
            if st.session_state.lambda_min_file is not None:
                new_config_min = max(st.session_state.lambda_min_file, substrate_min_wl)
                try:
                     # Check against current max before updating min
                    current_max = float(st.session_state.config_lambda_max) if st.session_state.config_lambda_max != "---" else np.inf
                    if new_config_min < current_max:
                        st.session_state.config_lambda_min = f"{new_config_min:.1f}"
                        add_log_message("info", f"Substrate changed to {st.session_state.substrate_choice}. Default Min Lambda updated to {st.session_state.config_lambda_min} nm.")
                    else:
                         add_log_message("warning", f"Substrate changed. New suggested min lambda ({new_config_min:.1f} nm) >= current max lambda ({st.session_state.config_lambda_max}). Manual adjustment needed.")
                except ValueError:
                     # Handle case where current_max is invalid
                     st.session_state.config_lambda_min = f"{new_config_min:.1f}"
                     add_log_message("info", f"Substrate changed to {st.session_state.substrate_choice}. Default Min Lambda updated to {st.session_state.config_lambda_min} nm.")

                st.rerun() # Rerun needed to update displayed value


        st.subheader("Optimization Lambda Range")
        col_lam1, col_lam2 = st.columns(2)
        with col_lam1:
            st.session_state.config_lambda_min = st.text_input("Min λ (nm):", value=st.session_state.config_lambda_min, help=f"Must be >= substrate limit ({get_substrate_min_lambda(st.session_state.substrate_choice):.1f} nm) and >= file min λ.")
        with col_lam2:
            st.session_state.config_lambda_max = st.text_input("Max λ (nm):", value=st.session_state.config_lambda_max, help="Must be <= file max λ.")

        # Lambda range validation hints
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

        if st.button("Plot Substrate Indices"):
            fig_sub = plot_substrate_indices()
            if fig_sub: st.session_state.fig_substrate_plot = fig_sub
            else: st.error("Could not generate substrate plot.")

        with st.expander("Advanced Optimization Settings"):
            adv_params = st.session_state.advanced_optim_params # Work with session state copy
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
            with col_mut2a: adv_params['mutation_max'] = st.number_input("Mutation Max", min_value=adv_params['mutation_min'], max_value=2.0, value=adv_params['mutation_max'], step=0.1) # Ensure max >= min
            adv_params['recombination'] = st.slider("Recombination", min_value=0.0, max_value=1.0, value=adv_params['recombination'], step=0.1)
            adv_params['polish'] = st.checkbox("Polish final solution", value=adv_params['polish'])
            strategy_options = ['best1bin', 'best1exp', 'rand1exp', 'randtobest1exp', 'currenttobest1exp', 'best2exp', 'rand2exp', 'randtobest1bin', 'currenttobest1bin', 'best2bin', 'rand2bin', 'rand1bin']
            adv_params['strategy'] = st.selectbox("DE Strategy", options=strategy_options, index=strategy_options.index(adv_params['strategy']))
            updating_options = ['immediate', 'deferred']
            adv_params['updating'] = st.selectbox("Updating Mode", options=updating_options, index=updating_options.index(adv_params['updating']))
            adv_params['workers'] = st.number_input("Parallel Workers (-1 = Auto, 1 = Off)", min_value=-1, value=adv_params['workers'], step=1, help="Uses multiple CPU cores if > 1 or -1. Set to 1 if pickling errors occur.")


        st.divider()
        if st.button("🔄 Reset Parameters"):
            # Reset basic parameters
            st.session_state.config_lambda_min = "---"
            st.session_state.config_lambda_max = "---"
            st.session_state.thickness_min = 300.0
            st.session_state.thickness_max = 600.0
            # Reset advanced parameters to default
            st.session_state.advanced_optim_params = default_advanced_params.copy()
            # If target data exists, try to reset lambda range based on it
            if st.session_state.target_data is not None:
                 substrate_min_wl_reset = get_substrate_min_lambda(st.session_state.substrate_choice)
                 initial_config_min_reset = max(st.session_state.lambda_min_file, substrate_min_wl_reset)
                 if initial_config_min_reset < st.session_state.lambda_max_file:
                    st.session_state.config_lambda_min = f"{initial_config_min_reset:.1f}"
                    st.session_state.config_lambda_max = f"{st.session_state.lambda_max_file:.1f}"

            add_log_message("info", "Configuration parameters reset to defaults.")
            st.rerun()


    # Main Area Columns
    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        st.subheader("Target Data")
        # Target Type Radio
        st.session_state.target_type = st.radio(
            "Select Target Type:",
            options=["T_norm", "T"],
            format_func=lambda x: "T Norm (%) = T_sample / T_sub" if x == "T_norm" else "T Sample (%)",
            index=["T_norm", "T"].index(st.session_state.target_type),
            horizontal=True,
            key="target_type_radio" # Key to potentially track changes
        )

        # File Uploader and Loading Logic
        uploaded_file = st.file_uploader(
            "Upload Target File (.csv) or use default:",
            type=["csv"],
            accept_multiple_files=False,
            key="file_uploader"
        )

        default_file_path = 'example.csv' # Define default file name
        data_source_to_load = None
        source_name = None
        is_default_file = False

        # Decide which source to load (new upload takes precedence)
        if uploaded_file is not None:
             # Check if it's a genuinely new file compared to last loaded
            if uploaded_file.name != st.session_state.get('last_loaded_source', None):
                data_source_to_load = uploaded_file
                source_name = uploaded_file.name
                st.session_state.last_loaded_source = source_name
                add_log_message("info", f"User uploaded file: {source_name}")
                # Reset results if new file loaded
                reset_log()
                st.session_state.optim_results = None
        elif st.session_state.target_data is None: # Only load default if no data currently loaded
            add_log_message("info", f"No file uploaded. Attempting to load default: {default_file_path}")
            if os.path.exists(default_file_path):
                # Check if default hasn't been loaded already in this session run
                if default_file_path != st.session_state.get('last_loaded_source', None):
                    data_source_to_load = default_file_path
                    source_name = default_file_path
                    st.session_state.last_loaded_source = source_name
                    is_default_file = True
                    # Reset results if default file loaded for first time
                    reset_log()
                    st.session_state.optim_results = None
            else:
                 # Only show warning once if default is missing
                if st.session_state.last_loaded_source != "DEFAULT_NOT_FOUND":
                    add_log_message("error", f"Default file '{default_file_path}' not found. Please upload a file.")
                    st.warning(f"Default file '{default_file_path}' not found. Please upload a file.", icon="⚠️")
                    st.session_state.last_loaded_source = "DEFAULT_NOT_FOUND"


        # File parsing logic (if a source was identified)
        if data_source_to_load is not None:
            st.session_state.target_data = None # Clear previous data before parsing
            try:
                file_extension = os.path.splitext(source_name)[1].lower()
                if file_extension != ".csv": raise ValueError(f"Unsupported file extension: '{file_extension}'. Please select a .csv file.")

                add_log_message("info", f"Parsing {source_name}...")
                df = None
                error_messages = []

                # Define parsing attempts (common CSV formats)
                parse_attempts = [
                    {'delimiter': ',', 'decimal': '.', 'encoding': 'utf-8', 'msg': "Parse attempt: Delimiter=',' Decimal='.' Encoding='utf-8'"},
                    {'delimiter': ';', 'decimal': '.', 'encoding': 'utf-8', 'msg': "Parse attempt: Delimiter=';' Decimal='.' Encoding='utf-8'"},
                    {'delimiter': '\t', 'decimal': '.', 'encoding': 'utf-8', 'msg': "Parse attempt: Delimiter='\\t' Decimal='.' Encoding='utf-8'"},
                    {'delimiter': ',', 'decimal': ',', 'encoding': 'latin-1', 'msg': "Parse attempt: Delimiter=',' Decimal=',' Encoding='latin-1'"},
                    {'delimiter': ';', 'decimal': ',', 'encoding': 'latin-1', 'msg': "Parse attempt: Delimiter=';' Decimal=',' Encoding='latin-1'"},
                ]

                for attempt in parse_attempts:
                    try:
                        add_log_message("info", attempt['msg'])
                        if hasattr(data_source_to_load, 'seek'): data_source_to_load.seek(0) # Ensure reading from start for file-like objects

                        # Use pandas read_csv with specific options
                        df_attempt = pd.read_csv(
                            data_source_to_load,
                            delimiter=attempt['delimiter'],
                            decimal=attempt['decimal'],
                            skiprows=1, # Assume header is present and skip it
                            usecols=[0, 1], # Expect data in first two columns
                            names=['lambda', 'target_value'], # Assign standard names
                            encoding=attempt['encoding'],
                            skipinitialspace=True,
                            on_bad_lines='warn' # Log bad lines instead of failing hard
                        )
                        # Basic validation after parsing
                        if df_attempt is not None and not df_attempt.empty and 'lambda' in df_attempt.columns and 'target_value' in df_attempt.columns:
                            # Try converting to numeric early to catch format errors specific to this attempt
                            lam_test = pd.to_numeric(df_attempt['lambda'], errors='coerce')
                            tgt_test = pd.to_numeric(df_attempt['target_value'], errors='coerce')
                            # Check if at least some values are numeric in both columns
                            if not lam_test.isnull().all() and not tgt_test.isnull().all():
                                df = df_attempt # Successful parse
                                add_log_message("info", f"Successfully parsed with: Delimiter='{attempt['delimiter']}' Decimal='{attempt['decimal']}' Encoding='{attempt['encoding']}'")
                                break # Stop trying other formats
                            else:
                                error_messages.append(f"Attempt ({attempt['msg']}) read columns but failed numeric conversion.")
                                add_log_message("warning", f"Parse attempt ({attempt['msg']}) failed numeric conversion.")
                        else:
                            error_messages.append(f"Attempt ({attempt['msg']}) failed to read expected columns or was empty.")
                            add_log_message("warning", f"Parse attempt ({attempt['msg']}) failed structure check.")

                    except Exception as e_parse:
                         error_messages.append(f"Attempt ({attempt['msg']}) failed: {e_parse}")
                         add_log_message("warning", f"Parse attempt ({attempt['msg']}) failed with exception: {e_parse}")


                if df is None:
                    raise ValueError(f"Could not read or parse the CSV file '{source_name}' after multiple attempts. Please check format (delimiter, decimal, encoding, header row). Errors: {'; '.join(error_messages)}")

                if df.empty: raise ValueError("CSV file loaded but appears empty or has incorrect structure after skipping header.")
                if 'lambda' not in df.columns or 'target_value' not in df.columns: raise ValueError("Could not find expected columns ('lambda', 'target_value'). Check header/columns.")

                # Convert to numeric, coercing errors to NaN
                df['lambda'] = pd.to_numeric(df['lambda'], errors='coerce'); df['target_value'] = pd.to_numeric(df['target_value'], errors='coerce')
                initial_rows = len(df)
                df.dropna(subset=['lambda', 'target_value'], inplace=True) # Remove rows with NaN in critical columns
                rows_after_na = len(df)
                if rows_after_na < initial_rows:
                    add_log_message("warning", f"{initial_rows - rows_after_na} rows dropped due to non-numeric values in lambda or target column.")

                if df.empty: raise ValueError("No valid numeric data rows found after parsing.")

                # Convert to numpy, sort by lambda
                data = df.to_numpy(); data = data[data[:, 0].argsort()]; lam = data[:, 0]; target_value_raw = data[:, 1]

                if len(lam) < 3: raise ValueError("Not enough valid data rows (< 3) after processing.") # Need points for analysis

                # Check lambda validity
                min_lam_val = np.min(lam)
                if min_lam_val <= 0: raise ValueError("Wavelengths (λ) must be > 0.")

                # Normalize target values if they look like percentages
                valid_targets = target_value_raw[np.isfinite(target_value_raw)]
                if len(valid_targets) > 0 and np.nanmax(valid_targets) > 5: # Heuristic: if max > 5, assume %
                    target_value_final = target_value_raw / 100.0
                    add_log_message("info", f"Target Data values > 5 detected, interpreting as % and dividing by 100.")
                else:
                    target_value_final = target_value_raw
                    add_log_message("info", f"Target Data values <= 5, interpreting as fraction (0-1).")


                # Store processed data in session state
                st.session_state.target_data = {'lambda': lam, 'target_value': target_value_final, 'target_type': st.session_state.target_type}
                st.session_state.lambda_min_file = np.min(lam); st.session_state.lambda_max_file = np.max(lam)
                if st.session_state.lambda_min_file >= st.session_state.lambda_max_file - SMALL_EPSILON: raise ValueError("Invalid wavelength range in file after processing (Min >= Max).")

                st.session_state.target_filename_base = source_name # Store base name for display

                # --- Update default Optimization Lambda Range based on loaded file ---
                current_substrate = st.session_state.substrate_choice; substrate_min_wl = get_substrate_min_lambda(current_substrate); initial_config_min = max(st.session_state.lambda_min_file, substrate_min_wl)
                # Ensure min < max
                if initial_config_min >= st.session_state.lambda_max_file:
                     add_log_message("warning", f"The minimum allowed wavelength ({initial_config_min:.1f} nm) for substrate {current_substrate} is >= max wavelength in file ({st.session_state.lambda_max_file:.1f} nm). Cannot set a valid default range.")
                     st.session_state.config_lambda_min = "---"; st.session_state.config_lambda_max = "---" # Indicate invalid range
                else:
                     st.session_state.config_lambda_min = f"{initial_config_min:.1f}"; st.session_state.config_lambda_max = f"{st.session_state.lambda_max_file:.1f}";
                     add_log_message("info", f"Set/Update default optimization range based on file and substrate ({current_substrate}): [{initial_config_min:.1f}, {st.session_state.lambda_max_file:.1f}] nm")


                # Plot the loaded target data immediately
                fig_target = plot_target_only(st.session_state.target_data, st.session_state.target_filename_base)
                if fig_target: st.session_state.fig_target_plot = fig_target # Store plot for display

                # Log success
                num_total_pts = rows_after_na
                log_source = "Default file" if is_default_file else "Uploaded file"
                add_log_message("info", f"{log_source} '{st.session_state.target_filename_base}' loaded ({num_total_pts} valid rows).")
                add_log_message("info", f"  λ range in file: [{st.session_state.lambda_min_file:.1f}, {st.session_state.lambda_max_file:.1f}] nm.")
                add_log_message("info", f"  Target type assumed: {st.session_state.target_type}.")

                st.rerun() # Rerun to update sidebar inputs and display plot

            except (ValueError, ImportError, Exception) as e:
                st.error(f"Error reading or processing file '{source_name}':\n{e}", icon="🚨"); add_log_message("error", f"ERROR loading file '{source_name}': {e}")
                # Reset state on error
                st.session_state.target_data = None; st.session_state.lambda_min_file = None; st.session_state.lambda_max_file = None;
                st.session_state.target_filename_base = "Error Loading"; st.session_state.config_lambda_min = "---"; st.session_state.config_lambda_max = "---"; st.session_state.fig_target_plot = None
                st.session_state.last_loaded_source = f"ERROR_{source_name}" # Avoid reload loop on error
                st.rerun()

        # Display Loaded File Info and Plot
        if st.session_state.target_filename_base:
             if st.session_state.target_filename_base == "Error Loading":
                 pass # Error message already shown
             elif st.session_state.target_data is not None:
                 file_source_msg = "(Default)" if st.session_state.last_loaded_source == default_file_path else "(Uploaded)"
                 st.success(f"Loaded: {st.session_state.target_filename_base} {file_source_msg} (Type: {st.session_state.target_type})", icon="✅")
                 # Display the plot stored in session state
                 if 'fig_target_plot' in st.session_state and st.session_state.fig_target_plot is not None:
                    st.pyplot(st.session_state.fig_target_plot)
                 else:
                     # This shouldn't happen if loading was successful, but as a fallback:
                     st.warning("Target data loaded, but plot could not be generated.", icon="⚠️")
        elif st.session_state.last_loaded_source != "DEFAULT_NOT_FOUND": # Avoid showing if default wasn't found
             st.info("Upload a CSV target file or ensure 'example.csv' is present for default loading.")


        # Thickness Config
        st.subheader("Thickness Configuration")
        col_thick1a, col_thick2a = st.columns(2)
        with col_thick1a: st.session_state.thickness_min = st.number_input("Min Thick (nm):", min_value=0.0, value=st.session_state.thickness_min, step=10.0, format="%.1f")
        with col_thick2a: st.session_state.thickness_max = st.number_input("Max Thick (nm):", min_value=max(0.1, st.session_state.thickness_min + 0.1), value=st.session_state.thickness_max, step=10.0, format="%.1f") # Ensure max > min

        st.divider()
        # Run Button
        run_button = st.button("▶ Run Optimization", type="primary", use_container_width=True, disabled=(st.session_state.target_data is None))


    with col2:
        # Schema Plot
        st.subheader("Comparative Schema")
        fig_schema = draw_schema_matplotlib(st.session_state.target_type, st.session_state.substrate_choice)
        st.pyplot(fig_schema)

        # Substrate Plot Display (if generated)
        if 'fig_substrate_plot' in st.session_state and st.session_state.fig_substrate_plot is not None:
            st.subheader("Substrate Indices")
            st.pyplot(st.session_state.fig_substrate_plot)
            if st.button("Clear Substrate Plot"):
                st.session_state.fig_substrate_plot = None;
                st.rerun() # Remove plot immediately


    # --- Optimization Execution Logic ---
    if run_button:
        reset_log(); st.session_state.optim_results = None
        add_log_message("info", "="*20 + " Starting Optimization " + "="*20)
        valid_params = True
        try:
            # --- Validate Parameters ---
            thickness_min = float(st.session_state.thickness_min); thickness_max = float(st.session_state.thickness_max)
            if thickness_min > thickness_max or thickness_min < 0: raise ValueError("Invalid Monolayer Thickness bounds (Min >= 0, Min <= Max).")

            lambda_min_str = st.session_state.config_lambda_min; lambda_max_str = st.session_state.config_lambda_max
            if not lambda_min_str or not lambda_max_str or lambda_min_str == "---" or lambda_max_str == "---": raise ValueError("Lambda Min/Max not set. Load a file and check sidebar configuration.")
            effective_lambda_min = float(lambda_min_str); effective_lambda_max = float(lambda_max_str)
            if effective_lambda_min <= 0 or effective_lambda_max <= 0: raise ValueError("Lambda Min/Max must be positive.")
            if effective_lambda_min >= effective_lambda_max: raise ValueError("Lambda Min must be less than Max Lambda.")

            # Check against file limits
            if st.session_state.lambda_min_file is None or st.session_state.lambda_max_file is None: raise ValueError("File lambda range not available. Load file again.")
            if effective_lambda_min < st.session_state.lambda_min_file - SMALL_EPSILON or effective_lambda_max > st.session_state.lambda_max_file + SMALL_EPSILON: raise ValueError(f"Optimization lambda range [{effective_lambda_min:.1f}, {effective_lambda_max:.1f}] nm must be within the loaded file's range [{st.session_state.lambda_min_file:.1f}, {st.session_state.lambda_max_file:.1f}] nm.")

            # Check against substrate limit
            selected_substrate = st.session_state.substrate_choice; substrate_min_limit = get_substrate_min_lambda(selected_substrate)
            if effective_lambda_min < substrate_min_limit:
                # Adjust effective_lambda_min upwards silently or warn user? Warn and adjust.
                 add_log_message("warning", f"Specified Min Lambda ({effective_lambda_min:.1f} nm) is below minimum for {selected_substrate} ({substrate_min_limit:.1f} nm). Adjusting Min Lambda to substrate limit.")
                 effective_lambda_min = substrate_min_limit
                 st.session_state.config_lambda_min = f"{effective_lambda_min:.1f}" # Update UI value
                 # Re-check if min >= max after adjustment
                 if effective_lambda_min >= effective_lambda_max: raise ValueError(f"Adjusted Min Lambda ({effective_lambda_min:.1f} nm) is now >= Max Lambda ({effective_lambda_max:.1f} nm). Check configuration.")
                 st.rerun() # Rerun to show adjusted value and prevent running with inconsistent state

            add_log_message("info", f"Using optimization range: [{effective_lambda_min:.1f}, {effective_lambda_max:.1f}] nm")
            current_target_type = st.session_state.target_type; target_type_flag = 0 if current_target_type == 'T_norm' else 1
            substrate_id = SUBSTRATE_LIST.index(selected_substrate)
            adv_params = st.session_state.advanced_optim_params; num_knots_n = adv_params['num_knots_n']; num_knots_k = adv_params['num_knots_k']; use_inv_lambda_sq_distrib = adv_params['use_inv_lambda_sq_distrib']
            # Validate knot counts
            if num_knots_n < 2 or num_knots_k < 2: raise ValueError("Number of n and k spline knots must be >= 2.")


        except (ValueError, TypeError, KeyError) as e_param: st.error(f"Parameter Error: Invalid parameter value:\n{e_param}\nCheck Thickness and Sidebar settings.", icon="🚨"); add_log_message("error", f"Parameter Error: {e_param}"); valid_params = False
        except Exception as e_unexpected: st.error(f"Setup Error: An unexpected error occurred during setup:\n{e_unexpected}", icon="🚨"); add_log_message("error", f"Unexpected Setup Error: {e_unexpected}"); valid_params = False


        if valid_params:
            # --- Setup before optimization ---
            add_log_message("info", f"Starting optimization: Target={current_target_type}, Substrate={selected_substrate}")
            add_log_message("info", f"Thickness Range: [{thickness_min:.1f}, {thickness_max:.1f}] nm"); add_log_message("info", f"Advanced parameters: {adv_params}")

            # Prepare data arrays for optimization
            current_target_lambda = st.session_state.target_data['lambda']; current_target_value = st.session_state.target_data['target_value']

            # --- Define mask for points used IN optimization ---
            # 1. Within user-defined lambda range
            lambda_range_mask = (current_target_lambda >= effective_lambda_min) & (current_target_lambda <= effective_lambda_max)
            # 2. Target value must be finite
            valid_target_mask_finite = np.isfinite(current_target_value) & np.isfinite(current_target_lambda)
            # 3. Substrate index must be calculable (i.e., lambda >= substrate min_wl)
            nSub_target_array_full = np.array([get_n_substrate(substrate_id, l) for l in current_target_lambda])
            valid_substrate_mask = np.isfinite(nSub_target_array_full)

            # Combine masks
            mask_used_in_optimization = valid_target_mask_finite & lambda_range_mask & valid_substrate_mask


            if not np.any(mask_used_in_optimization):
                st.error(f"No valid target data points found within the specified lambda range [{effective_lambda_min:.1f}, {effective_lambda_max:.1f}] nm with a valid substrate index.", icon="🚨"); add_log_message("error", "No valid points for optimization in the specified range.")
            else:
                # --- Prepare data ONLY for points used in optimization ---
                target_lambda_opt = current_target_lambda[mask_used_in_optimization]; target_value_opt = current_target_value[mask_used_in_optimization]; nSub_target_array_opt = nSub_target_array_full[mask_used_in_optimization]
                weights_array = np.ones_like(target_lambda_opt); # Currently using uniform weights
                add_log_message("info", f"Using {len(target_lambda_opt)} target points for optimization.")

                # --- Calculate Fixed Knot Positions ---
                fixed_n_knot_lambdas = np.array([], dtype=float); fixed_k_knot_lambdas = np.array([], dtype=float); knot_lam_min = effective_lambda_min; knot_lam_max = effective_lambda_max
                try:
                    if use_inv_lambda_sq_distrib:
                        # Use 1/lambda^2 distribution
                        inv_lambda_sq_min = 1.0 / (knot_lam_max**2); inv_lambda_sq_max = 1.0 / (knot_lam_min**2)
                        if num_knots_n > 0: fixed_n_knot_lambdas = 1.0 / np.sqrt(np.linspace(inv_lambda_sq_min, inv_lambda_sq_max, num_knots_n) + SMALL_EPSILON) # Add epsilon to avoid sqrt(0)
                        if num_knots_k > 0: fixed_k_knot_lambdas = 1.0 / np.sqrt(np.linspace(inv_lambda_sq_min, inv_lambda_sq_max, num_knots_k) + SMALL_EPSILON)
                        add_log_message("info", "Using 1/λ² knot distribution.")
                    else:
                        # Use 1/lambda distribution (default)
                        inv_lambda_min = 1.0 / knot_lam_max; inv_lambda_max = 1.0 / knot_lam_min
                        if num_knots_n > 0: fixed_n_knot_lambdas = 1.0 / (np.linspace(inv_lambda_min, inv_lambda_max, num_knots_n) + SMALL_EPSILON) # Avoid division by zero if linspace hits 0
                        if num_knots_k > 0: fixed_k_knot_lambdas = 1.0 / (np.linspace(inv_lambda_min, inv_lambda_max, num_knots_k) + SMALL_EPSILON)
                        add_log_message("info", "Using 1/λ knot distribution.")

                    # Ensure knots are sorted and within bounds (clip just in case of numerical issues)
                    fixed_n_knot_lambdas = np.clip(np.sort(fixed_n_knot_lambdas), knot_lam_min + SMALL_EPSILON, knot_lam_max - SMALL_EPSILON)
                    fixed_k_knot_lambdas = np.clip(np.sort(fixed_k_knot_lambdas), knot_lam_min + SMALL_EPSILON, knot_lam_max - SMALL_EPSILON)

                    # Check for duplicate knots which CubicSpline dislikes
                    if len(np.unique(fixed_n_knot_lambdas)) < num_knots_n or len(np.unique(fixed_k_knot_lambdas)) < num_knots_k:
                        add_log_message("warning", "Duplicate knot wavelengths generated, likely due to narrow range or low knot count. Adjusting slightly using linspace as fallback.")
                        # Fallback to simple linear spacing if duplicates occur
                        fixed_n_knot_lambdas = np.linspace(knot_lam_min + SMALL_EPSILON, knot_lam_max - SMALL_EPSILON, num_knots_n); fixed_k_knot_lambdas = np.linspace(knot_lam_min + SMALL_EPSILON, knot_lam_max - SMALL_EPSILON, num_knots_k)

                except Exception as e_knot: st.error(f"Knot Error: Error calculating knot positions for range [{knot_lam_min:.1f}, {knot_lam_max:.1f}] nm:\n{e_knot}", icon="🚨"); add_log_message("error", f"Knot calculation error: {e_knot}"); valid_params = False


                if valid_params:
                    # --- Define Optimization Bounds and Fixed Arguments ---
                    parameter_bounds = [(thickness_min, thickness_max)] + \
                                       [N_KNOT_VALUE_BOUNDS] * num_knots_n + \
                                       [LOG_K_KNOT_VALUE_BOUNDS] * num_knots_k
                    # Fixed arguments passed to the objective function
                    fixed_args = (num_knots_n, num_knots_k, target_lambda_opt, nSub_target_array_opt, target_value_opt, weights_array, target_type_flag, fixed_n_knot_lambdas, fixed_k_knot_lambdas)

                    # --- Setup Callback for Progress ---
                    optim_iteration_count = [0] # Use list to allow modification inside callback
                    optim_callback_best_mse = [np.inf] # Track best MSE found so far
                    status_text = st.empty() # Placeholder for status updates

                    def optimization_callback_simple_log(xk, convergence):
                        """Callback function to display optimization progress."""
                        optim_iteration_count[0] += 1
                        display_freq = 50 # How often to update the status message
                        is_best = False # Flag if this iteration improved the result
                        # Update status periodically or on first iteration
                        if optim_iteration_count[0] % display_freq == 0 or optim_iteration_count[0] == 1:
                            try:
                                # Evaluate objective function for current params (xk) to get current MSE
                                # Note: This re-calculates, might slow down slightly but gives real-time feedback
                                current_fun = objective_func_spline_fixed_knots(xk, *fixed_args)
                                if not np.isfinite(current_fun): current_fun = np.inf # Handle potential non-finite results

                                # Update best MSE seen so far
                                if current_fun < optim_callback_best_mse[0]:
                                    optim_callback_best_mse[0] = current_fun; is_best = True

                                mse_val = optim_callback_best_mse[0] # Display the best MSE found
                                status_text.info(f"Iteration: {optim_iteration_count[0]} | Best MSE: {mse_val:.4e}", icon="⏳")
                            except Exception as e_cb:
                                # Avoid crashing optimization due to callback error
                                add_log_message("warning", f"Error in callback at iter {optim_iteration_count[0]}: {e_cb}")


                    # --- Configure and Run Differential Evolution ---
                    de_args = {
                        'func': objective_func_spline_fixed_knots, 'bounds': parameter_bounds, 'args': fixed_args,
                        'strategy': adv_params['strategy'], 'maxiter': adv_params['maxiter'], 'popsize': adv_params['pop_size'],
                        'tol': adv_params['tol'], 'atol': adv_params['atol'], 'mutation': (adv_params['mutation_min'], adv_params['mutation_max']),
                        'recombination': adv_params['recombination'], 'polish': adv_params['polish'], 'updating': adv_params['updating'],
                        'workers': adv_params['workers'], # Use value from advanced params (could be -1, 1, or >1)
                        'disp': False, # Don't print to console, use callback
                        'callback': optimization_callback_simple_log
                       }

                    try:
                        with st.spinner("Optimization running... Please wait."):
                            start_time_opt = time.time();
                            optim_result = scipy.optimize.differential_evolution(**de_args);
                            end_time_opt = time.time()
                        status_text.success(f"Optimization finished in {end_time_opt - start_time_opt:.2f} s.", icon="✅"); add_log_message("info", f"Optimization finished in {end_time_opt - start_time_opt:.2f} s.")

                        if not optim_result.success: add_log_message("warning", f"Main optimization did not converge successfully: {optim_result.message}"); st.warning(f"Optimization warning: {optim_result.message}", icon="⚠️")


                        # --- Post-optimization Processing ---
                        add_log_message("info", "-" * 50); add_log_message("info", f"Best result from optimization:");
                        p_optimal = optim_result.x; final_objective_value = optim_result.fun; final_mse_display = final_objective_value if np.isfinite(final_objective_value) and final_objective_value < HUGE_PENALTY else np.nan
                        add_log_message("info", f"  Optimal MSE (Objective Func): {final_mse_display:.4e}")

                        # Extract optimal parameters
                        optimal_thickness_nm = p_optimal[0]; add_log_message("info", f"  Optimal Monolayer Thickness: {optimal_thickness_nm:.3f} nm");
                        idx_start = 1; n_values_opt_final = p_optimal[idx_start : idx_start + num_knots_n]; idx_start += num_knots_n; log_k_values_opt_final = p_optimal[idx_start : idx_start + num_knots_k]

                        # --- Recalculate Spectra & MSE over FULL original lambda range ---
                        # (Using the optimal parameters found)
                        add_log_message("info", "Recalculating spectra and MSE over full original lambda range...")
                        # Recreate final splines
                        n_spline_final = CubicSpline(fixed_n_knot_lambdas, n_values_opt_final, bc_type='natural', extrapolate=True); log_k_spline_final = CubicSpline(fixed_k_knot_lambdas, log_k_values_opt_final, bc_type='natural', extrapolate=True)
                        # Substrate index over full range (already calculated as nSub_target_array_full)

                        # Evaluate final n, k over full lambda range
                        n_final_array_recalc = np.full_like(current_target_lambda, np.nan); k_final_array_recalc = np.full_like(current_target_lambda, np.nan); T_stack_final_calc = np.full_like(current_target_lambda, np.nan); T_norm_final_calc = np.full_like(current_target_lambda, np.nan)

                        # Mask for where n/k can be calculated (spline eval)
                        valid_lambda_mask_for_calc = (current_target_lambda >= substrate_min_limit) & np.isfinite(current_target_lambda) & (current_target_lambda > 0)
                        if np.any(valid_lambda_mask_for_calc):
                             lambda_to_eval = current_target_lambda[valid_lambda_mask_for_calc]
                             n_final_array_recalc[valid_lambda_mask_for_calc] = n_spline_final(lambda_to_eval);
                             k_final_array_recalc[valid_lambda_mask_for_calc] = np.exp(log_k_spline_final(lambda_to_eval))
                             # Clip results to physical bounds after spline evaluation
                             n_final_array_recalc = np.clip(n_final_array_recalc, 1.0, N_KNOT_VALUE_BOUNDS[1]); # n>=1
                             k_final_array_recalc = np.clip(k_final_array_recalc, 0.0, math.exp(LOG_K_KNOT_VALUE_BOUNDS[1])) # k>=0
                             # Ensure NaN propagation if spline fails
                             n_final_array_recalc[~np.isfinite(n_final_array_recalc)] = np.nan; k_final_array_recalc[~np.isfinite(k_final_array_recalc)] = np.nan


                        # Calculate T_stack and T_norm where possible
                        valid_nk_final_mask = np.isfinite(n_final_array_recalc) & np.isfinite(k_final_array_recalc); valid_nsub_mask_recalc = np.isfinite(nSub_target_array_full); # Use full nSub array
                        valid_indices_for_T_calc = np.where(valid_nk_final_mask & valid_nsub_mask_recalc)[0]

                        for i in valid_indices_for_T_calc:
                             l_val = current_target_lambda[i]; nMono_val = n_final_array_recalc[i] - 1j * k_final_array_recalc[i]; nSub_val = nSub_target_array_full[i]
                             try:
                                 # Calculate T_stack and T_sub
                                 _, Ts_stack_calc, _ = calculate_monolayer_lambda(l_val, nMono_val, optimal_thickness_nm, nSub_val); _, Ts_sub_calc, _ = calculate_monolayer_lambda(l_val, 1.0 + 0j, 0.0, nSub_val)
                                 # Store T_stack if valid
                                 if np.isfinite(Ts_stack_calc): T_stack_final_calc[i] = np.clip(Ts_stack_calc, 0.0, 1.0)
                                 # Calculate and store T_norm if possible
                                 T_norm_calc = np.nan
                                 if np.isfinite(Ts_sub_calc):
                                     if Ts_sub_calc > SMALL_EPSILON: T_norm_calc = Ts_stack_calc / Ts_sub_calc
                                     elif abs(Ts_stack_calc) < SMALL_EPSILON : T_norm_calc = 0.0
                                 if np.isfinite(T_norm_calc): T_norm_final_calc[i] = np.clip(T_norm_calc, 0.0, 2.0)
                             except Exception: pass # Ignore errors in single point calculation during recalc


                        # --- Calculate Final MSE and Quality IN OPTIMIZATION RANGE ---
                        calc_value_for_mse = T_norm_final_calc if current_target_type == 'T_norm' else T_stack_final_calc;
                        # Use the original mask for points used in optimization
                        combined_valid_mask_for_mse = mask_used_in_optimization & np.isfinite(calc_value_for_mse) # Ensure calculated value is also finite

                        recalc_mse_final = np.nan; percent_good_fit = np.nan; quality_label = "N/A"; mse_pts_count = np.sum(combined_valid_mask_for_mse)
                        if mse_pts_count > 0 :
                             recalc_mse_final = np.mean((calc_value_for_mse[combined_valid_mask_for_mse] - current_target_value[combined_valid_mask_for_mse])**2);
                             # Quality assessment based on absolute difference within the opt range
                             abs_delta = np.abs(calc_value_for_mse[combined_valid_mask_for_mse] - current_target_value[combined_valid_mask_for_mse]); delta_threshold = 0.0025 # Threshold = 0.25%
                             points_below_threshold = np.sum(abs_delta < delta_threshold);
                             percent_good_fit = (points_below_threshold / mse_pts_count) * 100.0
                             # Assign quality label
                             if percent_good_fit >= 90: quality_label = "Excellent";
                             elif percent_good_fit >= 70: quality_label = "Good";
                             elif percent_good_fit >= 50: quality_label = "Fair";
                             else: quality_label = "Poor"
                             add_log_message("info", f"  Final MSE ({current_target_type}, {mse_pts_count} pts in range): {recalc_mse_final:.4e}"); add_log_message("info", "-"*20 + " Fit Quality " + "-"*20)
                             add_log_message("info", f"  Range [{effective_lambda_min:.1f}-{effective_lambda_max:.1f}] nm, {mse_pts_count} valid pts"); add_log_message("info", f"  Points with |delta| < {delta_threshold*100:.2f}% : {percent_good_fit:.1f}% ({points_below_threshold}/{mse_pts_count})"); add_log_message("info", f"  -> Rating: {quality_label}")
                        else:
                             add_log_message("warning", f"Cannot recalculate Final MSE or Fit Quality for range [{effective_lambda_min:.1f}-{effective_lambda_max:.1f}] nm. No valid points found after recalculation.")
                        add_log_message("info", "-" * 50)


                        # --- Store results in session state ---
                        plot_lambda_array_final = np.linspace(st.session_state.lambda_min_file, st.session_state.lambda_max_file, 500) # Use full file range for final nk plot if desired, or just opt range
                        # plot_lambda_array_final = np.linspace(effective_lambda_min, effective_lambda_max, 500) # Option: plot only opt range

                        st.session_state.optim_results = {
                            'final_spectra': { 'l': current_target_lambda, 'T_stack_calc': T_stack_final_calc, 'T_norm_calc': T_norm_final_calc,
                                               'MSE_Optimized': final_mse_display, 'MSE_Recalculated': recalc_mse_final,
                                               'percent_good_fit': percent_good_fit, 'quality_label': quality_label },
                            # Target data filtered only for plotting comparison (using optim mask?)
                            # 'target_filtered_for_plot': { 'lambda': current_target_lambda[mask_used_in_optimization], 'target_value': current_target_value[mask_used_in_optimization], 'target_type': current_target_type },
                            'best_params': { 'thickness_nm': optimal_thickness_nm, 'num_knots_n': num_knots_n, 'num_knots_k': num_knots_k,
                                             'n_knot_values': n_values_opt_final, 'log_k_knot_values': log_k_values_opt_final,
                                             'n_knot_lambdas': fixed_n_knot_lambdas, 'k_knot_lambdas': fixed_k_knot_lambdas,
                                             'knot_distribution': "1/λ²" if use_inv_lambda_sq_distrib else "1/λ", 'substrate_name': selected_substrate,
                                             'effective_lambda_min': effective_lambda_min, 'effective_lambda_max': effective_lambda_max },
                            'plot_lambda_array': plot_lambda_array_final, # Wavelengths for nk plot
                            'model_str_base': "Spline Fit",
                            # Data for table display (full range)
                            'result_data_table': {
                                'lambda (nm)': current_target_lambda,
                                f'n (Spline Fit ({num_knots_n}n{num_knots_k}k))': n_final_array_recalc,
                                f'k (Spline Fit ({num_knots_n}n{num_knots_k}k))': k_final_array_recalc,
                                'Thickness (nm)': np.full_like(current_target_lambda, optimal_thickness_nm), # Constant value
                                f'n Substrate ({selected_substrate})': nSub_target_array_full,
                                f'Target {current_target_type} (%) (Used)': np.where(mask_used_in_optimization, current_target_value * 100.0, np.nan), # Show only target used
                                f'Target {current_target_type} (%) (Full File)': current_target_value * 100.0, # Show original full target
                                'Calc T (%)': T_stack_final_calc * 100.0,
                                'Calc T Norm (%)': T_norm_final_calc * 100.0,
                                'Delta T (%)': (T_stack_final_calc - current_target_value)*100.0 if current_target_type == 'T' else np.nan,
                                'Delta T Norm (%)': (T_norm_final_calc - current_target_value)*100.0 if current_target_type == 'T_norm' else np.nan,
                             }
                           }

                        # --- Removed Excel file generation block ---

                    except Exception as e_optim: st.error(f"Optimization Error: An error occurred during optimization:\n{e_optim}", icon="🚨"); add_log_message("error", f"ERROR during optimization: {e_optim}"); traceback.print_exc()


    # --- Results Display ---
    if st.session_state.optim_results:
        results = st.session_state.optim_results
        st.divider(); st.header("Optimization Results")
        # Display metrics
        col_res1a, col_res2a = st.columns(2)
        with col_res1a: st.metric("Optimal Thickness", f"{results['best_params']['thickness_nm']:.3f} nm")
        with col_res2a:
            mse_disp = results['final_spectra']['MSE_Recalculated'];
            st.metric("Final MSE (in range)", f"{mse_disp:.4e}" if np.isfinite(mse_disp) else "N/A")

        quality_label = results['final_spectra']['quality_label']
        percent_good_fit = results['final_spectra']['percent_good_fit']
        if np.isfinite(percent_good_fit):
            st.metric("Fit Quality Rating (in range)", f"{quality_label}", help=f"Based on {percent_good_fit:.1f}% points within optim. range having |Calc - Target| < 0.25%")
        else:
            st.metric("Fit Quality Rating (in range)", "N/A")


        st.subheader("Result Plots")
        # Plotting - Comparison Spectrum
        fig_compare = plot_spectra_vs_target(
            res=results['final_spectra'],
            target=st.session_state.target_data, # Pass the full original target data for plotting
            best_params_info=results['best_params'],
            model_str_base=results['model_str_base'],
            effective_lambda_min=results['best_params']['effective_lambda_min'],
            effective_lambda_max=results['best_params']['effective_lambda_max']
        )
        if fig_compare: st.pyplot(fig_compare)

        # Plotting - Final n/k
        fig_nk = plot_nk_final(results['best_params'], results['plot_lambda_array'])
        if fig_nk: st.pyplot(fig_nk)


        st.subheader("Result Data")
        # --- Removed Email/Download Button ---
        with st.expander("Show Result Data Table (Full Range)"):
             # Use the stored data table dictionary
            if 'result_data_table' in results:
                try:
                    df_display = pd.DataFrame(results['result_data_table'])#.set_index('lambda (nm)')
                    # Apply formatting for display
                    formatters = {col: "{:.4f}" for col in df_display.columns if df_display[col].dtype == 'float64'}
                    # Override format for percentage columns
                    percent_cols = [col for col in df_display.columns if '%' in col]
                    for pcol in percent_cols: formatters[pcol] = "{:.2f}"
                    # Override format for thickness
                    formatters['Thickness (nm)'] = "{:.3f}"

                    st.dataframe(df_display.style.format(formatters, na_rep='-'))
                except Exception as e_df:
                    st.warning(f"Could not display result table: {e_df}")
            else:
                st.warning("Result data table not available for display.")


    # --- Display User Log Section ---
    with st.expander("Show User Access Log"):
        display_user_log()


    # --- Log Display ---
    display_log()

    # --- Help Text ---
    help_text_en = """
    User Manual - Optical Monolayer Optimizer (Streamlit Version)

    Goal:
    This program determines the optical properties (refractive index n, extinction coefficient k) and thickness (d) of a single thin film (monolayer) deposited on a known substrate. It adjusts these parameters so that the calculated transmission best matches experimental target data.

    Main Steps:

    1.  **Provide Your Email:** Enter your email address (and optionally name) on the initial screen. This email is logged for usage tracking.
    2.  **Configure Settings (Sidebar):**
        * **Substrate Material:** Choose the substrate. The valid wavelength range depends on this.
        * **Optimization Lambda Range:** Define the Min/Max wavelength (nm) for the fitting process. Must be within file range and substrate valid range.
        * **Advanced Optimization Settings (Optional):** Fine-tune spline knots, DE parameters (population, iterations, tolerance, strategy), and parallelism (`workers`). Set `workers=1` to disable parallel execution if pickling errors occur.
        * **Reset Parameters:** Reset settings to default.

    3.  **Select Target Type (Main Area):**
        * Choose "Normalized Transmission" (T Norm (%)) = T_sample / T_sub, or "Sample Transmission" (T Sample (%)). Schema updates accordingly.

    4.  **Load Target File (Main Area):**
        * Upload a **.csv** file or use the default `example.csv` if present.
        * **CSV Format:** Expects a header row (skipped), then Column 1: λ (nm), Column 2: Target value. Delimiter (`,` or `;` or `\t`) and decimal (`.` or `,`) are auto-detected.
        * Target values are interpreted as fractions (0-1) or percentages (automatically divided by 100 if > 5). A plot of the loaded data appears.

    5.  **Configure Thickness Range (Main Area):**
        * Define **Min/Max Thick (nm)** for the optimization search.

    6.  **Run Optimization (Main Area):**
        * Click "▶ Run Optimization". Progress (iteration and best MSE) is shown.

    7.  **Analyze Results (Main Area):**
        * **Metrics:** Optimal Thickness, Final MSE (calculated within the optimization range), Fit Quality Rating (% points with delta < 0.25%).
        * **Result Plots:**
            * Comparison plot: Target data (points), Calculated spectrum (line), and Delta (Calc-Target, dashed line on right axis, shown *only* within optimization range).
            * Final n/k plot: Optimized n (blue line) and k (red dashed line) with knot positions marked.
        * **Result Data:** An expander ("Show Result Data Table") allows viewing the calculated n, k, T values alongside target data for the full wavelength range of the input file.
    8.  **User Access Log:** An expander ("Show User Access Log") displays a list of users who have accessed the application (timestamp, name, email).

    Tips:
    - Check CSV format (header, columns, delimiter, decimal) and Log Messages if loading fails.
    - Choose realistic thickness and Lambda ranges. Ensure Lambda range is valid for the chosen substrate.
    - Visually inspect plots for physical plausibility (smooth n/k, Kramers-Kronig consistency hinted) alongside mathematical fit quality (MSE, Quality Rating).
    - Numba caching (`cache=True`) is enabled to potentially improve stability with multiprocessing. If pickling errors persist, set `workers=1` in Advanced Settings.
    - The User Access Log persistence depends on the deployment environment (may be lost on restarts in some cloud platforms).
    """


    with st.expander("Help / Instructions", expanded=False):
        st.markdown(help_text_en)

    # Sidebar Footer
    st.sidebar.markdown("---")
    st.sidebar.info("Monolayer Optimizer v1.4 - Streamlit App adapted from original code by F. Lemarchand.")
