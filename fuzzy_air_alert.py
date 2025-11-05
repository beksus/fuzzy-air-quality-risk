#!/usr/bin/env python3
"""
fuzzy_air_alert.py

Complete implementation of the Fuzzy Inference System for Public Health Alert Level
based on the PDF specification with all 5 inputs, comprehensive rule base, AND visual outputs.
"""

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import os

# ------------------------
# 1) Build Complete FIS with ALL 5 inputs
# ------------------------

# Input variables with proper ranges based on WHO/EPA standards
pm25 = ctrl.Antecedent(np.arange(0, 251, 1), 'pm25')  # μg/m³
no2 = ctrl.Antecedent(np.arange(0, 201, 1), 'no2')  # ppb
o3 = ctrl.Antecedent(np.arange(0, 201, 1), 'o3')  # ppb
wind_speed = ctrl.Antecedent(np.arange(0, 21, 0.1), 'wind_speed')  # m/s
pvi = ctrl.Antecedent(np.arange(0, 11, 0.1), 'pvi')  # Population Vulnerability Index 0-10

# Output variable
alert = ctrl.Consequent(np.arange(0, 101, 1), 'alert')  # Public Health Alert Level 0-100

# ------------------------
# 2) Membership Functions based on report specifications
# ------------------------

# PM2.5 Membership Functions (based on Kho Weng Khai's guidelines)
pm25['low'] = fuzz.trapmf(pm25.universe, [0, 0, 10, 25])
pm25['moderate'] = fuzz.trimf(pm25.universe, [15, 25, 35])
pm25['high'] = fuzz.trimf(pm25.universe, [30, 45, 60])
pm25['very_high'] = fuzz.trapmf(pm25.universe, [55, 70, 250, 250])

# NO2 Membership Functions
no2['low'] = fuzz.trapmf(no2.universe, [0, 0, 40, 80])
no2['moderate'] = fuzz.trimf(no2.universe, [50, 80, 110])
no2['high'] = fuzz.trapmf(no2.universe, [90, 120, 200, 200])

# O3 Membership Functions
o3['low'] = fuzz.trapmf(o3.universe, [0, 0, 50, 80])
o3['moderate'] = fuzz.trimf(o3.universe, [60, 90, 120])
o3['high'] = fuzz.trapmf(o3.universe, [100, 130, 200, 200])

# Wind Speed Membership Functions (from PDF Table 2)
wind_speed['low'] = fuzz.trapmf(wind_speed.universe, [0, 0, 1, 2.5])
wind_speed['medium'] = fuzz.trimf(wind_speed.universe, [1.5, 4, 8])
wind_speed['high'] = fuzz.trapmf(wind_speed.universe, [6, 8, 20, 20])

# Population Vulnerability Index Membership Functions
pvi['low'] = fuzz.trapmf(pvi.universe, [0, 0, 2, 4])
pvi['medium'] = fuzz.trimf(pvi.universe, [3, 5, 7])
pvi['high'] = fuzz.trapmf(pvi.universe, [6, 8, 10, 10])

# Alert Level Membership Functions (matching PDF output categories)
alert['low_risk'] = fuzz.trapmf(alert.universe, [0, 0, 15, 25])
alert['moderate_advisory'] = fuzz.trimf(alert.universe, [20, 35, 50])
alert['high_alert'] = fuzz.trimf(alert.universe, [45, 60, 75])
alert['severe_warning'] = fuzz.trapmf(alert.universe, [70, 85, 100, 100])

# ------------------------
# 3) Comprehensive Rule Base (Representative subset of 324 possible rules)
# ------------------------

rules = [
    # Category 1: Baseline Rules
    ctrl.Rule(pm25['low'] & no2['low'] & o3['low'], alert['low_risk']),
    ctrl.Rule(pm25['very_high'], alert['severe_warning']),
    ctrl.Rule(no2['high'] & pm25['low'] & o3['low'], alert['high_alert']),
    ctrl.Rule(o3['high'] & pm25['low'] & no2['low'], alert['high_alert']),

    # Category 2: Synergistic-Effect Rules
    ctrl.Rule(pm25['moderate'] & o3['moderate'], alert['moderate_advisory']),
    ctrl.Rule(pm25['moderate'] & no2['moderate'] & o3['moderate'], alert['high_alert']),
    ctrl.Rule(pm25['moderate'] & o3['moderate'] & wind_speed['low'], alert['high_alert']),
    ctrl.Rule(pm25['high'] & no2['moderate'] & wind_speed['low'], alert['severe_warning']),

    # Category 3: Vulnerability-Amplifying Rules
    ctrl.Rule(pm25['moderate'] & pvi['high'], alert['high_alert']),
    ctrl.Rule(no2['high'] & pvi['high'], alert['severe_warning']),
    ctrl.Rule(o3['moderate'] & pvi['high'], alert['high_alert']),
    ctrl.Rule(pm25['high'] & pvi['high'], alert['severe_warning']),
    ctrl.Rule(no2['high'] & pvi['high'], alert['severe_warning']),
    ctrl.Rule(pm25['low'] & pvi['high'], alert['moderate_advisory']),

    # Category 4: Mitigation Rules
    ctrl.Rule(o3['high'] & wind_speed['high'], alert['moderate_advisory']),
    ctrl.Rule(pm25['high'] & wind_speed['high'], alert['moderate_advisory']),
    ctrl.Rule(no2['high'] & wind_speed['high'], alert['moderate_advisory']),

    # Additional synergistic rules
    ctrl.Rule(pm25['high'] & no2['moderate'], alert['high_alert']),
    ctrl.Rule(pm25['moderate'] & no2['high'], alert['high_alert']),
    ctrl.Rule(pm25['high'] & o3['moderate'], alert['high_alert']),
]

# ------------------------
# 4) Create and compile the control system
# ------------------------

system = ctrl.ControlSystem(rules)
sim = ctrl.ControlSystemSimulation(system)


# ------------------------
# 5) Enhanced Sample Dataset with ALL 5 variables
# ------------------------

def create_sample_dataset():
    """Create realistic sample data with all 5 input variables"""
    rng = np.random.default_rng(42)

    # 30 days of sample data
    n_days = 30

    # Create segments for different pollution scenarios
    segments = [
        ('clean', 8, 15, 5),  # 8 clean days
        ('moderate', 10, 40, 8),  # 10 moderate days
        ('high', 7, 80, 15),  # 7 high days
        ('very_high', 5, 120, 25)  # 5 very high days
    ]

    pm25_vals = []
    no2_vals = []
    o3_vals = []

    for scenario, count, mean, std in segments:
        pm25_segment = np.clip(rng.normal(mean, std, count), 5, 240)
        pm25_vals.extend(pm25_segment)

        # Generate correlated NO2 and O3 values
        no2_segment = np.clip(pm25_segment * 0.8 + rng.normal(30, 20, count), 10, 190)
        no2_vals.extend(no2_segment)

        o3_segment = np.clip(rng.normal(mean * 0.7, std, count), 15, 190)
        o3_vals.extend(o3_segment)

    # Trim to exact n_days in case of rounding
    pm25_vals = pm25_vals[:n_days]
    no2_vals = no2_vals[:n_days]
    o3_vals = o3_vals[:n_days]

    # Wind speed (often inversely correlated with pollution)
    wind_vals = np.clip(15 - (np.array(pm25_vals) / 20) + rng.normal(0, 2, n_days), 0.5, 18)

    # Population Vulnerability Index (some areas consistently more vulnerable)
    pvi_segments = [
        ('low_vuln', 10, 3, 1),  # 10 low vulnerability days
        ('medium_vuln', 12, 6, 1.5),  # 12 medium vulnerability days
        ('high_vuln', 8, 9, 1)  # 8 high vulnerability days
    ]

    pvi_vals = []
    for scenario, count, mean, std in pvi_segments:
        pvi_segment = np.clip(rng.normal(mean, std, count), 0, 10)
        pvi_vals.extend(pvi_segment)

    pvi_vals = pvi_vals[:n_days]

    df = pd.DataFrame({
        'date': pd.date_range(start='2025-06-01', periods=n_days, freq='D').astype(str),
        'pm25': np.round(pm25_vals, 1),
        'no2': np.round(no2_vals, 1),
        'o3': np.round(o3_vals, 1),
        'wind_speed': np.round(wind_vals, 1),
        'pvi': np.round(pvi_vals, 1)
    })

    return df


# ------------------------
# 6) PLOTTING FUNCTIONS (from fyzzy_air_alert_v2.py)
# ------------------------

def save_membership_plots(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    ax = axes.flatten()

    # PM2.5
    ax[0].plot(pm25.universe, pm25['low'].mf, 'b', linewidth=1.5, label='Low')
    ax[0].plot(pm25.universe, pm25['moderate'].mf, 'g', linewidth=1.5, label='Moderate')
    ax[0].plot(pm25.universe, pm25['high'].mf, 'orange', linewidth=1.5, label='High')
    ax[0].plot(pm25.universe, pm25['very_high'].mf, 'r', linewidth=1.5, label='Very High')
    ax[0].set_title('PM2.5 Membership Functions')
    ax[0].legend()
    ax[0].set_ylabel('Membership')
    ax[0].set_xlabel('PM2.5 (μg/m³)')
    ax[0].grid(True, alpha=0.3)

    # NO2
    ax[1].plot(no2.universe, no2['low'].mf, 'b', linewidth=1.5, label='Low')
    ax[1].plot(no2.universe, no2['moderate'].mf, 'g', linewidth=1.5, label='Moderate')
    ax[1].plot(no2.universe, no2['high'].mf, 'r', linewidth=1.5, label='High')
    ax[1].set_title('NO2 Membership Functions')
    ax[1].legend()
    ax[1].set_ylabel('Membership')
    ax[1].set_xlabel('NO2 (ppb)')
    ax[1].grid(True, alpha=0.3)

    # O3
    ax[2].plot(o3.universe, o3['low'].mf, 'b', linewidth=1.5, label='Low')
    ax[2].plot(o3.universe, o3['moderate'].mf, 'g', linewidth=1.5, label='Moderate')
    ax[2].plot(o3.universe, o3['high'].mf, 'r', linewidth=1.5, label='High')
    ax[2].set_title('O3 Membership Functions')
    ax[2].legend()
    ax[2].set_ylabel('Membership')
    ax[2].set_xlabel('O3 (ppb)')
    ax[2].grid(True, alpha=0.3)

    # Wind Speed
    ax[3].plot(wind_speed.universe, wind_speed['low'].mf, 'b', linewidth=1.5, label='Low')
    ax[3].plot(wind_speed.universe, wind_speed['medium'].mf, 'g', linewidth=1.5, label='Medium')
    ax[3].plot(wind_speed.universe, wind_speed['high'].mf, 'orange', linewidth=1.5, label='High')
    ax[3].set_title('Wind Speed Membership Functions')
    ax[3].legend()
    ax[3].set_ylabel('Membership')
    ax[3].set_xlabel('Wind Speed (m/s)')
    ax[3].grid(True, alpha=0.3)

    # PVI
    ax[4].plot(pvi.universe, pvi['low'].mf, 'b', linewidth=1.5, label='Low')
    ax[4].plot(pvi.universe, pvi['medium'].mf, 'g', linewidth=1.5, label='Medium')
    ax[4].plot(pvi.universe, pvi['high'].mf, 'r', linewidth=1.5, label='High')
    ax[4].set_title('Population Vulnerability Index Membership Functions')
    ax[4].legend()
    ax[4].set_ylabel('Membership')
    ax[4].set_xlabel('PVI (0-10)')
    ax[4].grid(True, alpha=0.3)

    # Alert Level
    ax[5].plot(alert.universe, alert['low_risk'].mf, 'g', linewidth=1.5, label='Low Risk')
    ax[5].plot(alert.universe, alert['moderate_advisory'].mf, 'y', linewidth=1.5, label='Moderate Advisory')
    ax[5].plot(alert.universe, alert['high_alert'].mf, 'orange', linewidth=1.5, label='High Alert')
    ax[5].plot(alert.universe, alert['severe_warning'].mf, 'r', linewidth=1.5, label='Severe Warning')
    ax[5].set_title('Public Health Alert Level Membership Functions')
    ax[5].legend()
    ax[5].set_ylabel('Membership')
    ax[5].set_xlabel('Alert Level (0-100)')
    ax[5].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'membership_functions_complete.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    return path


def save_timeseries_plot(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    try:
        df['date_parsed'] = pd.to_datetime(df['date'])
    except Exception:
        df['date_parsed'] = pd.RangeIndex(len(df))

    raw = df['fuzzy_alert_index'].astype(float)
    plt.figure(figsize=(12, 6))

    # Create color-coded scatter plot based on alert level
    colors = {
        'Low Risk': 'green',
        'Moderate Advisory': 'yellow',
        'High Alert': 'orange',
        'Severe Warning': 'red'
    }

    for level, color in colors.items():
        mask = df['alert_level'] == level
        if mask.any():
            plt.scatter(df.loc[mask, 'date_parsed'], df.loc[mask, 'fuzzy_alert_index'],
                        c=color, label=level, s=50, alpha=0.7)

    # Connect points with lines
    plt.plot(df['date_parsed'], raw, 'k-', alpha=0.3, linewidth=1)

    # Highlight min and max
    min_idx = raw.idxmin()
    max_idx = raw.idxmax()
    plt.scatter([df['date_parsed'].iloc[min_idx]], [raw.iloc[min_idx]],
                color='blue', s=100, marker='*', label='Minimum', zorder=5)
    plt.scatter([df['date_parsed'].iloc[max_idx]], [raw.iloc[max_idx]],
                color='purple', s=100, marker='*', label='Maximum', zorder=5)

    plt.title(
        f'Fuzzy Public Health Alert Level Over Time\n(Min: {raw.min():.1f}, Max: {raw.max():.1f}, Mean: {raw.mean():.1f})')
    plt.xlabel('Date')
    plt.ylabel('Public Health Alert Level (0-100)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    timeseries_path = os.path.join(out_dir, 'fuzzy_alert_timeseries_complete.png')
    plt.savefig(timeseries_path, dpi=200, bbox_inches='tight')
    plt.close()
    return timeseries_path


def save_alert_distribution_plot(df, out_dir):
    """Additional plot: Distribution of alert levels"""
    os.makedirs(out_dir, exist_ok=True)

    alert_counts = df['alert_level'].value_counts().sort_index()
    colors = ['green', 'yellow', 'orange', 'red']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(alert_counts.index.astype(str), alert_counts.values, color=colors, alpha=0.7)

    plt.title('Distribution of Public Health Alert Levels')
    plt.xlabel('Alert Level')
    plt.ylabel('Number of Days')
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)} days', ha='center', va='bottom')

    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    dist_path = os.path.join(out_dir, 'alert_level_distribution.png')
    plt.savefig(dist_path, dpi=200, bbox_inches='tight')
    plt.close()
    return dist_path


# ------------------------
# 7) Run FIS for all rows
# ------------------------

def run_fuzzy_analysis(input_csv=None, output_dir='output'):
    """Main function to run the complete fuzzy analysis"""

    # Load or create dataset
    if input_csv and os.path.isfile(input_csv):
        df = pd.read_csv(input_csv)
        # Normalize column names
        df.columns = [c.strip().lower().replace(' ', '_').replace('.', '') for c in df.columns]
    else:
        df = create_sample_dataset()
        print(f"Created sample dataset with {len(df)} rows")

    # Ensure numeric data
    for col in ['pm25', 'no2', 'o3', 'wind_speed', 'pvi']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna().reset_index(drop=True)

    results = []
    fired_rules_log = []

    def evaluate_rules(pm25_val, no2_val, o3_val, wind_val, pvi_val):
        """Evaluate which rules fire strongly for a given input set"""
        fired = []
        if pm25_val <= 25 and no2_val <= 80 and o3_val <= 80:
            fired.append("R1: Clean air -> Low Risk")
        if pm25_val >= 75:
            fired.append("R2: Very high PM2.5 -> Severe Warning")
        if 25 <= pm25_val <= 55 and 60 <= o3_val <= 120 and wind_val <= 2.5:
            fired.append("R4: Moderate PM2.5+O3 + Low wind -> High Alert")
        if 25 <= pm25_val <= 55 and pvi_val >= 6:
            fired.append("R5: Moderate PM2.5 + High PVI -> High Alert")
        if no2_val >= 100 and pvi_val >= 6:
            fired.append("R7: High NO2 + High PVI -> Severe Warning")
        return fired

    print("Processing data through fuzzy inference system...")

    for i, row in df.iterrows():
        try:
            # Set all 5 inputs
            sim.input['pm25'] = float(row['pm25'])
            sim.input['no2'] = float(row['no2'])
            sim.input['o3'] = float(row['o3'])
            sim.input['wind_speed'] = float(row['wind_speed'])
            sim.input['pvi'] = float(row['pvi'])

            # Compute output
            sim.compute()
            out = float(sim.output['alert'])

            # Log which rules fired
            fired_rules = evaluate_rules(
                row['pm25'], row['no2'], row['o3'],
                row['wind_speed'], row['pvi']
            )

        except Exception as e:
            out = np.nan
            fired_rules = [f"Error: {e}"]
            print(f"Computation failed on row {i}: {e}")

        results.append(out)
        fired_rules_log.append(fired_rules)

    df['fuzzy_alert_index'] = np.round(results, 2)
    df['fired_rules'] = fired_rules_log

    # Map to categorical labels
    df['alert_level'] = pd.cut(
        df['fuzzy_alert_index'],
        bins=[-0.1, 25, 50, 75, 100],
        labels=['Low Risk', 'Moderate Advisory', 'High Alert', 'Severe Warning']
    )

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, 'air_quality_complete_analysis.csv')
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")

    # Generate plots
    mf_path = save_membership_plots(output_dir)
    ts_path = save_timeseries_plot(df, output_dir)
    dist_path = save_alert_distribution_plot(df, output_dir)

    print(f"Saved membership functions: {mf_path}")
    print(f"Saved timeseries plot: {ts_path}")
    print(f"Saved distribution plot: {dist_path}")

    return df


# ------------------------
# 8) Printing out in terminal the comprehensive report
# ------------------------

def generate_report(df):
    """Generate the comprehensive analysis report"""

    print("\n" + "=" * 60)
    print("FUZZY AIR QUALITY ALERT SYSTEM - COMPLETE ANALYSIS")
    print("=" * 60)

    print(f"\nDataset Overview:")
    print(f"Total days analyzed: {len(df)}")
    if 'date' in df.columns:
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    print("\nPollutant Statistics:")
    print(df[['pm25', 'no2', 'o3']].describe().round(1))

    print("\nEnvironmental & Vulnerability Factors:")
    print(df[['wind_speed', 'pvi']].describe().round(1))

    print("\nFuzzy Alert Level Distribution:")
    alert_counts = df['alert_level'].value_counts().sort_index()
    for level, count in alert_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {level}: {count} days ({percentage:.1f}%)")

    # Traditional AQI comparison
    def calculate_simple_aqi(pm25_val, no2_val, o3_val):
        def pm25_to_aqi(pm25):
            if pm25 <= 12:
                return (pm25 / 12) * 50
            elif pm25 <= 35.4:
                return 50 + ((pm25 - 12.1) / (35.4 - 12.1)) * 50
            elif pm25 <= 55.4:
                return 100 + ((pm25 - 35.5) / (55.4 - 35.5)) * 50
            elif pm25 <= 150.4:
                return 150 + ((pm25 - 55.5) / (150.4 - 55.5)) * 100
            else:
                return 300 + ((pm25 - 150.5) / (250.4 - 150.5)) * 100

        def no2_to_aqi(no2):
            if no2 <= 53:
                return (no2 / 53) * 50
            elif no2 <= 100:
                return 50 + ((no2 - 54) / (100 - 54)) * 50
            elif no2 <= 360:
                return 100 + ((no2 - 101) / (360 - 101)) * 100
            else:
                return 200

        def o3_to_aqi(o3):
            if o3 <= 54:
                return (o3 / 54) * 50
            elif o3 <= 70:
                return 50 + ((o3 - 55) / (70 - 55)) * 50
            elif o3 <= 85:
                return 100 + ((o3 - 71) / (85 - 71)) * 50
            elif o3 <= 105:
                return 150 + ((o3 - 86) / (105 - 86)) * 100
            else:
                return 250

        aqi_pm25 = min(pm25_to_aqi(pm25_val), 500)
        aqi_no2 = min(no2_to_aqi(no2_val), 500)
        aqi_o3 = min(o3_to_aqi(o3_val), 500)

        return max(aqi_pm25, aqi_no2, aqi_o3)

    df['traditional_aqi'] = df.apply(
        lambda row: calculate_simple_aqi(row['pm25'], row['no2'], row['o3']), axis=1
    )

    df['aqi_vs_fuzzy_diff'] = df['traditional_aqi'] - df['fuzzy_alert_index']

    print("\n" + "=" * 50)
    print("COMPARISON: Traditional AQI vs Fuzzy System")
    print("=" * 50)

    high_discrepancy = df[df['aqi_vs_fuzzy_diff'].abs() > 20]
    print(f"Days with significant difference (>20 points): {len(high_discrepancy)}")

    print(f"\n✅ Analysis complete! All outputs saved to 'output' directory.")

    return df


# ------------------------
# 9) Main execution
# ------------------------

if __name__ == '__main__':
    # Create output directory
    output_dir = 'output'

    # Run the complete analysis
    df = run_fuzzy_analysis(output_dir=output_dir)

    # Generate comprehensive report
    generate_report(df)

    # Show sample results
    print(f"\nFirst 5 days of analysis:")
    display_cols = ['date', 'pm25', 'no2', 'o3', 'wind_speed', 'pvi', 'fuzzy_alert_index', 'alert_level']
    available_cols = [col for col in display_cols if col in df.columns]
    print(df[available_cols].head().to_string(index=False))