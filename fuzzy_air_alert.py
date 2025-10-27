#!/usr/bin/env python3
"""
fuzzy_air_alert.py

Complete Fuzzy Inference System for "Public Health Alert Level" from urban air-quality CSVs.

- Inputs: PM2.5, NO2, O3 (µg/m3), Vulnerability (0..1), Wind speed (m/s)
- Output: fuzzy_alert_index (0..100) and alert_level (Low, Moderate, High-sensitive, Severe-all)

Usage:
    python fuzzy_air_alert.py --input /mnt/data/kuala-lumpur-air-quality.csv \
                              --output /mnt/data/kuala-lumpur-air-quality-with-alerts.csv

Requirements:
    pip install numpy pandas matplotlib
"""

import argparse
import os
from math import isclose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Membership function helpers
# -----------------------------
def trapmf(x, a, b, c, d):
    """Trapezoidal MF"""
    x = np.array(x, dtype=float)
    y = np.zeros_like(x)
    # rising edge
    idx = (a < x) & (x <= b)
    if not isclose(b, a):
        y[idx] = (x[idx] - a) / (b - a)
    else:
        y[idx] = 1.0
    # top
    idx2 = (b < x) & (x <= c)
    y[idx2] = 1.0
    # falling edge
    idx3 = (c < x) & (x < d)
    if not isclose(d, c):
        y[idx3] = (d - x[idx3]) / (d - c)
    else:
        y[idx3] = 1.0
    return y

def trimf(x, a, b, c):
    """Triangular MF via trapezoid with top at b"""
    return trapmf(x, a, b, b, c)

# -----------------------------
# Universe definitions & MFs
# -----------------------------
u_pm25 = np.linspace(0, 300, 301)
u_no2  = np.linspace(0, 300, 301)
u_o3   = np.linspace(0, 300, 301)
u_vul  = np.linspace(0, 1, 101)
u_wind = np.linspace(0, 10, 101)
u_out  = np.linspace(0, 100, 101)

# PM2.5
pm25_low  = trapmf(u_pm25, 0, 0, 12, 25)
pm25_mod  = trimf(u_pm25, 12, 35, 55)
pm25_high = trimf(u_pm25, 35, 75, 125)
pm25_vhigh= trapmf(u_pm25, 75, 125, 300, 300)

# NO2
no2_low  = trapmf(u_no2, 0, 0, 40, 80)
no2_mod  = trimf(u_no2, 40, 100, 160)
no2_high = trimf(u_no2, 100, 180, 240)
no2_vhigh= trapmf(u_no2, 180, 240, 300, 300)

# O3
o3_low  = trapmf(u_o3, 0, 0, 30, 60)
o3_mod  = trimf(u_o3, 30, 80, 130)
o3_high = trimf(u_o3, 80, 150, 220)
o3_vhigh= trapmf(u_o3, 150, 220, 300, 300)

# Vulnerability
vul_low  = trapmf(u_vul, 0.0, 0.0, 0.2, 0.4)
vul_med  = trimf(u_vul, 0.2, 0.5, 0.8)
vul_high = trapmf(u_vul, 0.6, 0.8, 1.0, 1.0)

# Wind speed (low => poor dispersion)
wind_low = trapmf(u_wind, 0.0, 0.0, 1.0, 3.0)
wind_mod = trimf(u_wind, 1.5, 4.0, 6.5)
wind_high= trapmf(u_wind, 5.0, 7.0, 10.0, 10.0)

# Output MFs
out_low   = trapmf(u_out, 0, 0, 15, 30)
out_mod   = trimf(u_out, 20, 37.5, 55)
out_highs = trimf(u_out, 45, 62.5, 80)
out_severe= trapmf(u_out, 70, 82.5, 100, 100)

input_mfs = {
    'pm25': {'low': pm25_low, 'mod': pm25_mod, 'high': pm25_high, 'vhigh': pm25_vhigh},
    'no2' : {'low': no2_low,  'mod': no2_mod,  'high': no2_high,  'vhigh': no2_vhigh},
    'o3'  : {'low': o3_low,   'mod': o3_mod,   'high': o3_high,   'vhigh': o3_vhigh},
    'vul' : {'low': vul_low,  'med': vul_med,  'high': vul_high},
    'wind': {'low': wind_low, 'mod': wind_mod, 'high': wind_high},
}

output_mfs = {'low': out_low, 'moderate': out_mod, 'highs': out_highs, 'severe': out_severe}

# -----------------------------
# Rule Base
# -----------------------------
# Format: (conditions_dict, output_label, weight)
rules = [
    ({'pm25':'vhigh', 'vul':'high'}, 'severe', 1.0),
    ({'no2':'vhigh',  'vul':'high'}, 'severe', 1.0),
    ({'o3':'vhigh',   'vul':'high'}, 'severe', 1.0),
    ({'pm25':'high', 'no2':'high', 'wind':'low'}, 'severe', 0.95),
    ({'pm25':'high', 'o3':'high',  'wind':'low'}, 'severe', 0.95),
    ({'no2':'high',  'o3':'high',  'wind':'low'}, 'severe', 0.95),
    ({'pm25':'high', 'vul':['med','low']}, 'highs', 0.9),
    ({'no2':'high',  'vul':['med','low']}, 'highs', 0.9),
    ({'o3':'high',   'vul':['med','low']}, 'highs', 0.9),
    ({'pm25':'mod', 'no2':'mod', 'o3':'mod', 'wind':['mod','high']}, 'moderate', 0.8),
    ({'pm25':'mod', 'vul':'low', 'wind':'high'}, 'moderate', 0.7),
    ({'pm25':'low', 'no2':'low', 'o3':'low'}, 'low', 1.0),
    ({'vul':'low', 'wind':'high'}, 'low', 0.8),
    ({'pm25':['mod','high'], 'vul':'high'}, 'highs', 0.7),
    ({'no2':['mod','high'],  'vul':'high'}, 'highs', 0.7),
    ({'o3':['mod','high'],   'vul':'high'}, 'highs', 0.7),
]

# -----------------------------
# FIS functions
# -----------------------------
def fuzzify_value(value, universe, mfs_dict):
    memberships = {}
    if pd.isna(value):
        for lbl in mfs_dict:
            memberships[lbl] = 0.0
        return memberships
    for label, mf in mfs_dict.items():
        idx = np.abs(universe - float(value)).argmin()
        memberships[label] = float(mf[idx])
    return memberships

def eval_rule(rule, fuzzies):
    conds, out_label, weight = rule
    degrees = []
    for var, required in conds.items():
        if isinstance(required, list):
            # OR across listed labels
            degrees.append(max(fuzzies[var].get(r, 0.0) for r in required))
        else:
            degrees.append(fuzzies[var].get(required, 0.0))
    firing_strength = min(degrees) * weight
    return out_label, firing_strength

def aggregate_rule_outputs(firing_results):
    agg = {label: np.zeros_like(u_out) for label in output_mfs.keys()}
    for label, strength in firing_results:
        if strength <= 0:
            continue
        mf = output_mfs[label]
        clipped = np.minimum(mf, strength)
        agg[label] = np.maximum(agg[label], clipped)
    combined = np.zeros_like(u_out)
    for label in agg:
        combined = np.maximum(combined, agg[label])
    return agg, combined

def defuzz_centroid(x, mf):
    denom = mf.sum()
    if isclose(denom, 0.0):
        return float(np.nan)
    return float((x * mf).sum() / denom)

def run_fis_single(sample):
    """
    sample: dict with keys 'pm25','no2','o3','vul','wind'
    returns: dict with 'crisp' (index 0..100), fuzzies, firing, agg_by_label, combined
    """
    fuzzies = {}
    fuzzies['pm25'] = fuzzify_value(sample.get('pm25', np.nan), u_pm25, input_mfs['pm25'])
    fuzzies['no2']  = fuzzify_value(sample.get('no2', np.nan),  u_no2,  input_mfs['no2'])
    fuzzies['o3']   = fuzzify_value(sample.get('o3', np.nan),   u_o3,   input_mfs['o3'])
    fuzzies['vul']  = fuzzify_value(sample.get('vul', np.nan),  u_vul,  input_mfs['vul'])
    fuzzies['wind'] = fuzzify_value(sample.get('wind', np.nan), u_wind, input_mfs['wind'])

    firing = []
    for rule in rules:
        out_label, strength = eval_rule(rule, fuzzies)
        firing.append((out_label, strength))

    agg_by_label, combined = aggregate_rule_outputs(firing)
    crisp_out = defuzz_centroid(u_out, combined)
    return {'fuzzies': fuzzies, 'firing': firing, 'agg_by_label': agg_by_label, 'combined': combined, 'crisp': crisp_out}

# -----------------------------
# CSV processing & plotting
# -----------------------------
def normalize_column_names(df):
    new_cols = []
    for c in df.columns:
        nc = c.strip().lower().replace('.', '').replace(' ', '').replace('-', '')
        new_cols.append(nc)
    df.columns = new_cols
    return df

def map_columns(df):
    col_map = {}
    for c in df.columns:
        if 'pm25' in c or 'pm2' in c:
            col_map[c] = 'pm25'
        elif 'no2' in c:
            col_map[c] = 'no2'
        elif 'o3' in c:
            col_map[c] = 'o3'
        elif 'vul' in c or 'vulnerability' in c:
            col_map[c] = 'vul'
        elif 'wind' in c:
            col_map[c] = 'wind'
        elif 'date' in c:
            col_map[c] = 'date'
    return df.rename(columns=col_map)

def process_csv(input_path, output_path, default_vul=0.6, default_wind=3.0, save_plots=True, plots_dir=None):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    df = normalize_column_names(df)
    df = map_columns(df)

    # Convert pollutant columns to numeric and clean blank strings
    df = df.replace(r'^\s*$', np.nan, regex=True)
    for col in ['pm25', 'no2', 'o3']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Check required columns
    missing = [c for c in ['pm25','no2','o3'] if c not in df.columns]
    if missing:
        raise KeyError(f"CSV missing required pollutant columns: {missing}. Found: {df.columns.tolist()}")

    # Drop rows with missing pollutant data (optional: change to imputation)
    df_clean = df.dropna(subset=['pm25','no2','o3']).copy()

    # Ensure vulnerability & wind are present, else fill defaults
    if 'vul' not in df_clean.columns:
        df_clean['vul'] = default_vul
    else:
        df_clean['vul'] = pd.to_numeric(df_clean['vul'], errors='coerce').fillna(default_vul)

    if 'wind' not in df_clean.columns:
        df_clean['wind'] = default_wind
    else:
        df_clean['wind'] = pd.to_numeric(df_clean['wind'], errors='coerce').fillna(default_wind)

    # Run FIS row-wise
    results = []
    for _, row in df_clean.iterrows():
        sample = {'pm25': row['pm25'], 'no2': row['no2'], 'o3': row['o3'], 'vul': row['vul'], 'wind': row['wind']}
        res = run_fis_single(sample)
        results.append(res['crisp'])

    df_clean['fuzzy_alert_index'] = results
    df_clean['alert_level'] = pd.cut(df_clean['fuzzy_alert_index'], bins=[-1,25,50,75,100],
                                     labels=['Low','Moderate','High-sensitive','Severe-all'])

    # Save CSV
    df_clean.to_csv(output_path, index=False)
    print(f"Saved processed CSV with alerts to: {output_path}")

    # Optional plots directory
    if save_plots:
        if plots_dir is None:
            plots_dir = os.path.dirname(output_path) or '.'
        os.makedirs(plots_dir, exist_ok=True)
        save_mf_plots(plots_dir)
        # try parse date for timeseries
        try:
            df_clean['date_parsed'] = pd.to_datetime(df_clean['date'], infer_datetime_format=True)
        except Exception:
            df_clean['date_parsed'] = pd.RangeIndex(start=0, stop=len(df_clean))
        # timeseries plot
        plt.figure(figsize=(10, 4))
        raw = df_clean['fuzzy_alert_index']
        scaled = (raw - raw.min()) / (raw.max() - raw.min()) * 100

        plt.plot(df_clean['date_parsed'], raw, 'o-', label='Raw index')
        plt.plot(df_clean['date_parsed'], scaled, 'r--', label='Scaled (0–100)')
        plt.title('Fuzzy Alert Index over time (raw vs scaled)')
        plt.xlabel('Date')
        plt.ylabel('Fuzzy Alert Index')
        plt.legend()

        plt.tight_layout()
        timeseries_path = os.path.join(plots_dir, 'fuzzy_alert_timeseries.png')
        plt.savefig(timeseries_path, dpi=200)
        plt.close()
        print(f"Saved time-series plot to: {timeseries_path}")


    # return processed dataframe
    return df_clean

def save_mf_plots(out_dir):
    # Save a combined plot of input & output MFs
    fig, axes = plt.subplots(3,2, figsize=(12,12))
    ax = axes.flatten()

    ax[0].plot(u_pm25, pm25_low); ax[0].plot(u_pm25, pm25_mod); ax[0].plot(u_pm25, pm25_high); ax[0].plot(u_pm25, pm25_vhigh)
    ax[0].set_title('PM2.5 MFs'); ax[0].legend(['low','mod','high','vhigh'])
    ax[1].plot(u_no2, no2_low); ax[1].plot(u_no2, no2_mod); ax[1].plot(u_no2, no2_high); ax[1].plot(u_no2, no2_vhigh)
    ax[1].set_title('NO2 MFs'); ax[1].legend(['low','mod','high','vhigh'])
    ax[2].plot(u_o3, o3_low); ax[2].plot(u_o3, o3_mod); ax[2].plot(u_o3, o3_high); ax[2].plot(u_o3, o3_vhigh)
    ax[2].set_title('O3 MFs'); ax[2].legend(['low','mod','high','vhigh'])
    ax[3].plot(u_vul, vul_low); ax[3].plot(u_vul, vul_med); ax[3].plot(u_vul, vul_high)
    ax[3].set_title('Vulnerability MFs'); ax[3].legend(['low','med','high'])
    ax[4].plot(u_wind, wind_low); ax[4].plot(u_wind, wind_mod); ax[4].plot(u_wind, wind_high)
    ax[4].set_title('Wind MFs'); ax[4].legend(['low','mod','high'])
    ax[5].plot(u_out, out_low); ax[5].plot(u_out, out_mod); ax[5].plot(u_out, out_highs); ax[5].plot(u_out, out_severe)
    ax[5].set_title('Output Alert Level MFs'); ax[5].legend(['low','moderate','highs','severe'])
    plt.tight_layout()
    path = os.path.join(out_dir, 'membership_functions.png')
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved membership function plot to: {path}")

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Fuzzy Air Quality → Public Health Alert Level')
    p.add_argument('--input', '-i', required=True, help='Input CSV path')
    p.add_argument('--output', '-o', required=True, help='Output CSV path (with alerts)')
    p.add_argument('--vul-default', type=float, default=0.6, help='Default vulnerability (0..1) if not in CSV')
    p.add_argument('--wind-default', type=float, default=3.0, help='Default wind speed (m/s) if not in CSV')
    p.add_argument('--plots-dir', default=None, help='Directory to save plots (defaults to output dir)')
    p.add_argument('--no-plots', action='store_true', help='Do not save plots')
    return p.parse_args()

def main():
    args = parse_args()
    plots_dir = args.plots_dir if args.plots_dir else None
    save_plots = not args.no_plots
    df_processed = process_csv(args.input, args.output, default_vul=args.vul_default,
                               default_wind=args.wind_default, save_plots=save_plots, plots_dir=plots_dir)
    # Print brief counts
    counts = df_processed['alert_level'].value_counts().reindex(['Low','Moderate','High-sensitive','Severe-all']).fillna(0).astype(int)
    print("Alert level counts:")
    print(counts.to_string())

if __name__ == '__main__':
    input_path = r"C:\Users\xd\PycharmProjects\PythonProject1\kuala-lumpur-air-quality.csv"  # <-- your real path
    output_path = r"C:\Users\xd\PycharmProjects\PythonProject1\air-alert-results.csv"       # output file path

    process_csv(input_path, output_path, default_vul=0.6, default_wind=3.0, save_plots=True)
