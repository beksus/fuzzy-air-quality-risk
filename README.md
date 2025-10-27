# Fuzzy Air Quality Risk Assessment

### Overview
This project implements a **Fuzzy Inference System (FIS)** to evaluate **public health risks** from urban air quality, addressing **SDG 11** (Sustainable Cities and Communities) and **SDG 3** (Good Health and Well-being).

### Problem Definition
Traditional Air Quality Index (AQI) values simplify pollution into a single score. This can mask health risks caused by multiple moderate pollutants or the increased vulnerability of specific populations.  
This project applies fuzzy logic to model the **Public Health Alert Level** more precisely.

### Features
- Inputs: PM2.5, NO₂, O₃, Population Vulnerability, Wind Speed  
- Output: Fuzzy Alert Index (0–100) and Alert Level (Low → Severe)  
- Rule Base: Combines pollutant concentration and vulnerability using IF–THEN logic  
- Defuzzification: Centroid method (Mamdani inference)  
- Plots: Membership functions and time-series of alert indices  

### Tools & Libraries
- Python 3.11+
- Numpy, Pandas, Matplotlib

### Run
```bash
python fuzzy_air_alert.py --input "kuala-lumpur-air-quality.csv" --output "results.csv"
