Fuzzy Air Quality Alert System - Kuala Lumpur
A comprehensive fuzzy logic-based decision support system for urban air pollution public health risk assessment, specifically designed for Kuala Lumpur air quality data from AQICN.org.

Table of Contents
Overview

Data Source

Key Features

System Architecture

Installation

Usage

Input Variables

Output

Fuzzy Logic Implementation

Results

Project Structure

References

Overview
This project implements a Mamdani-type Fuzzy Inference System (FIS) that provides nuanced public health risk assessments for Kuala Lumpur by integrating multiple environmental and demographic factors. Unlike traditional AQI systems that use a simple "maximum operator," this system considers:

Multiple pollutant interactions

Meteorological conditions

Population vulnerability factors

Synergistic effects between variables

The system directly supports Sustainable Development Goals (SDG) 3 (Good Health and Well-being) and SDG 11 (Sustainable Cities and Communities).

Data Source
Primary Data: AQICN.org - Kuala Lumpur
Website: https://aqicn.org/city/kuala-lumpur/

The system uses real air quality data from Kuala Lumpur, including:

PM2.5 - Fine particulate matter concentrations

PM10 - Coarse particulate matter

O3 - Ozone levels

NO2 - Nitrogen dioxide

SO2 - Sulfur dioxide

CO - Carbon monoxide

Data Characteristics
Temporal Resolution: Daily measurements

Pollutant Coverage: Multiple parameters simultaneously

Geographic Focus: Kuala Lumpur urban area

Data Quality: Real-time monitoring with historical archives

Data Integration
For complete analysis, the system supplements AQICN data with:

Wind Speed: Meteorological data for pollutant dispersion

Population Vulnerability Index (PVI): Demographic and health indicators

Key Features
Kuala Lumpur Specific: Tailored for local air quality patterns

Multi-factor Integration: Combines 5 input variables for comprehensive risk assessment

Population Vulnerability Index: Considers demographic and health factors specific to KL

Meteorological Adaptation: Accounts for tropical wind dispersion effects

Transparent Rule-based Logic: 17 fuzzy rules across 4 logical categories

Comparative Analysis: Shows advantages over traditional AQI

Visual Analytics: Generates membership function plots and timeseries charts

Real-time Capable: Designed for practical deployment in KL

System Architecture
Input Variables
Variable	Unit	Range	Linguistic Terms	Data Source
PM2.5	μg/m³	0-250	Low, Moderate, High, Very High	AQICN.org
NO2	ppb	0-200	Low, Moderate, High	AQICN.org
O3	ppb	0-200	Low, Moderate, High	AQICN.org
Wind Speed	m/s	0-20	Low, Medium, High	Meteorological Data
Population Vulnerability Index	0-10	0-10	Low, Medium, High	Demographic Data
Output Variable
Public Health Alert Level (0-100 scale)

0-25: Low Risk

26-50: Moderate Advisory

51-75: High Alert for Sensitive Groups

76-100: Severe Warning for All

Installation
Prerequisites
Python 3.8+

pip package manager

Dependencies
bash
pip install numpy pandas scikit-fuzzy matplotlib
Quick Setup
bash
# Clone or download the project files
git clone <repository-url>
cd fuzzy-air-alert

# Install requirements
pip install -r requirements.txt
Usage
Using Kuala Lumpur Data
Download data from AQICN.org:

Visit: https://aqicn.org/city/kuala-lumpur/

Export historical data as CSV

Save as kuala-lumpur-air-quality.csv

Run the analysis:

bash
python fuzzy_air_alert_complete_with_plots.py
Input Data Format (AQICN Export)
Your CSV should include these columns:

csv
date,pm25,pm10,o3,no2,so2,co,aqi
2025/10/1,65.0,30,28.0,8.0,...,...
2025/10/2,75.0,32,7.0,8.0,...,...
Custom Data Analysis
bash
# Using your own CSV data
python fuzzy_air_alert_complete_with_plots.py --input your_data.csv --output results/
🔬 Fuzzy Logic Implementation
Membership Functions
Triangular and Trapezoidal functions for simplicity and interpretability

Parameters based on WHO and Malaysia DOE standards

Linguistic variables for natural reasoning

Tuned for KL conditions considering tropical climate

Rule Categories
Baseline Rules: Handle clean air and single high pollutants

Synergistic-Effect Rules: Capture combined pollutant risks

Vulnerability-Amplifying Rules: Escalate risk for vulnerable populations in KL

Mitigation Rules: Account for tropical wind dispersion effects

Kuala Lumpur Specific Considerations
python
# Accounting for KL's urban heat island effect
IF pm25 is High AND wind_speed is Low THEN alert is High_Alert

# Considering traffic patterns (major NO2 source)
IF no2 is High AND pm25 is Moderate THEN alert is Severe_Warning

# Vulnerability in dense urban areas
IF pvi is High AND pm25 is Moderate THEN alert is High_Alert
Results
Sample Output from KL Data
text
FUZZY AIR QUALITY ALERT SYSTEM - KUALA LUMPUR ANALYSIS
============================================================

Dataset Overview:
Total days analyzed: 91
Date range: 2025/09/21 to 2025/10/1

Pollutant Statistics:
        pm25    no2    o3
count   91.0   91.0  91.0
mean    68.9    7.8  24.1
std      5.1    0.8   6.2

Fuzzy Alert Level Distribution:
  High Alert: 81 days (89.0%)
  Severe Warning: 10 days (11.0%)
  Moderate Advisory: 0 days (0.0%)
  Low Risk: 0 days (0.0%)
Key Insights from KL Data
Consistent High Pollution: Majority of days show elevated risk levels

PM2.5 Dominance: Primary driver of health risks in Kuala Lumpur

Seasonal Patterns: Haze episodes significantly increase severe warnings

Urban Vulnerability: Dense population amplifies public health impacts

Generated Visualizations
membership_functions_complete.png - All input/output variable fuzzy sets

fuzzy_alert_timeseries_complete.png - Alert levels over time in KL

alert_level_distribution.png - Risk category distribution

air_quality_complete_analysis.csv - Detailed results data

Kuala Lumpur Specific Analysis
Pollution Patterns
Primary Concern: PM2.5 from vehicular emissions and seasonal haze

Secondary Pollutants: O3 formation in urban environment

Geographic Factors: Valley topography affecting pollutant dispersion

Temporal Trends: Higher pollution during dry seasons and rush hours

Public Health Implications
Vulnerable Groups: Children, elderly, and outdoor workers

High Density Areas: Greater KL region with population concentration

Economic Impact: Healthcare costs and productivity losses

Environmental Justice: Equitable risk distribution across communities

Project Structure
text
fuzzy-air-quality_risk/
├── fuzzy_air_alert.py  # Main implementation
├── kuala-lumpur-air-quality.csv            # AQICN.org data
├── output/                                 # Generated outputs
│   ├── membership_functions_complete.png
│   ├── fuzzy_alert_timeseries_complete.png
│   ├── alert_level_distribution.png
│   └── air_quality_complete_analysis.csv
└── README.md
Academic Background
Problem Statement
Traditional AQI systems suffer from:

Single-pollutant focus (maximum operator limitation)

Ignored population vulnerability in dense urban areas

No meteorological consideration for tropical climates

Binary risk thresholds unsuitable for gradual pollution changes

KL-Specific Challenges
Haze episodes from regional biomass burning

Traffic congestion contributing to NO2 and PM2.5

Urban heat island effect on pollutant formation

High population density increasing exposure risks

Performance Advantages for KL
vs Traditional AQI
Better risk discrimination during haze episodes

Early warnings for sensitive populations before AQI thresholds

More nuanced alerts considering KL's unique pollution mix

Improved equity considering urban density variations

Key Benefits for Kuala Lumpur
Proactive health warnings before severe pollution events

Targeted advisories for different city zones

Integration with KL health services for preparedness

Public awareness through transparent risk communication

Future Enhancements for KL
Real-time integration with DOE Malaysia monitoring stations

Mobile application for KL residents with location-based alerts

Seasonal adjustment for monsoon and haze patterns

Traffic data integration for source attribution

Hospital admissions correlation for validation

References
AQICN.org - Kuala Lumpur real-time air quality data

Department of Environment Malaysia - Air Quality Standards

UN Sustainable Development Goals - SDG 3 and SDG 11

World Health Organization - Air Quality Guidelines

Clarke et al. (2022) - Social Vulnerability Index for Air Pollution

Cromar et al. (2020) - Evaluating AQI as Risk Communication Tool

Contributors
Beksultan Kirgizbaev 22078406
Darrence Beh Heng Shek 23094907
Kho Weng Khai xxxxxxxx

Computational Intelligence Course Project

Data Source: AQICN.org - Kuala Lumpur

📄 License
This project is for academic and research purposes. Please cite appropriately if used in research. Data from AQICN.org is used under their terms of service.
