# Fuzzy Air Quality Alert System — Kuala Lumpur

**A fuzzy logic-based decision support system for urban air pollution public health risk assessment**, designed for Kuala Lumpur using air-quality data from AQICN.org.

---

## Table of Contents
- [Overview](#overview)  
- [Data Source](#data-source)  
- [Key Features](#key-features)  
- [System Architecture](#system-architecture)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Input Variables](#input-variables)  
- [Output](#output)  
- [Fuzzy Logic Implementation](#fuzzy-logic-implementation)  
- [Results (sample)](#results-sample)  
- [Project Structure](#project-structure)  
- [References](#references)  
- [Contributors & License](#contributors--license)

---

## Overview
This project implements a **Mamdani-type Fuzzy Inference System (FIS)** that produces a nuanced Public Health Alert Level (PHAL) for Kuala Lumpur by combining multiple pollutant measures, meteorological data, and a Population Vulnerability Index (PVI). It complements traditional AQI by accounting for multi-pollutant interactions, dispersion conditions, and demographic vulnerability — supporting **SDG 3 (Good Health)** and **SDG 11 (Sustainable Cities)**.

---

## Data Source
**Primary:** AQICN.org — Kuala Lumpur  
URL: https://aqicn.org/city/kuala-lumpur/

Typical data fields used (AQICN export):
- `date`, `pm25`, `pm10`, `o3`, `no2`, `so2`, `co`, `aqi`

Additional inputs used or synthesized:
- `wind_speed` (m/s) — meteorological dispersion factor  
- `pvi` (0–10) — Population Vulnerability Index (demographic/health indicator)

---

## Key Features
- KL-tailored fuzzy model with 5 inputs (PM2.5, NO₂, O₃, wind, PVI)
- Mamdani FIS with triangular & trapezoidal membership functions
- Representative rule base organized into Baseline / Synergistic / Vulnerability / Mitigation categories
- Visual outputs: membership functions, time-series, distribution plots
- Comparative analysis: fuzzy index vs. traditional AQI
- Designed for reproducibility (CSV + PNG outputs)

---

## System Architecture

**Inputs**
| Variable | Unit | Range | Linguistic Terms |
|---|---:|---:|---|
| PM2.5 | µg/m³ | 0–250 | Low / Moderate / High / Very High |
| NO₂ | ppb | 0–200 | Low / Moderate / High |
| O₃ | ppb | 0–200 | Low / Moderate / High |
| Wind Speed | m/s | 0–20 | Low / Medium / High |
| PVI | — | 0–10 | Low / Medium / High |

**Output**
- Public Health Alert Level (0–100)
  - 0–25: **Low Risk**
  - 26–50: **Moderate Advisory**
  - 51–75: **High Alert (sensitive groups)**
  - 76–100: **Severe Warning (all)**

---

## Installation

Requires Python 3.8+.

```bash
# clone or copy repo
git clone https://github.com/beksus/fuzzy-air-quality-risk.git
cd fuzzy-air-quality-risk

# optional: create virtual env
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
