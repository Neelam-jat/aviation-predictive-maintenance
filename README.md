<div align="center">

# ✈️ Aviation Predictive Maintenance

**AI-Driven Aircraft Engine Health Monitoring and Remaining Useful Life (RUL) Prediction**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)](https://github.com/)

</div>

---
## 📖 About the Project (The "Why")

In the aviation industry, unexpected engine failures lead to catastrophic safety risks, massive financial losses, and widespread flight delays. Historically, airlines have relied on **Scheduled (Preventive) Maintenance**, replacing expensive engine components after a fixed number of flights, regardless of their actual condition. This results in millions of dollars wasted on perfectly healthy parts.

**This project introduces a Predictive Maintenance pipeline.** 

By analyzing high-frequency sensor data (temperature, pressure, vibration) from aircraft turbofan engines, this Machine Learning model accurately predicts the **Remaining Useful Life (RUL)** of an engine. Instead of replacing parts based on a manual, airlines can replace them exactly when they are about to fail—maximizing operational efficiency, optimizing the supply chain, and ensuring absolute passenger safety.

> **Note to Recruiters & Engineers:** This project transforms raw, noisy aerospace data into actionable business intelligence, bridging the gap between raw data engineering and high-value machine learning.

---

## ✨ Key Features

* **Intelligent Target Calculation:** Implements a highly realistic **Piecewise RUL** strategy, capping maximum engine life to focus the AI strictly on the critical degradation phase.
* **Automated Feature Engineering:** Dynamically identifies and removes "flatline" (constant) sensors that offer no predictive value, heavily reducing computational noise.
* **Robust Machine Learning Architecture:** Utilizes an optimized **Random Forest Regressor** to capture complex, non-linear relationships across 21 different engine sensors.
* **Scalable Data Pipeline:** Built to seamlessly ingest, clean, and process multi-variate time-series data from the renowned NASA CMAPSS dataset.

---

## ⚙️ How It Works (Under the Hood)

The core logic of this project is broken down into a four-step pipeline:

1. **Data Ingestion:** Loads the NASA CMAPSS simulation data. Each row represents a single flight cycle containing 21 internal sensor readings for a specific engine.
2. **Target Generation (Piecewise RUL):** 
   - *The Problem:* A brand-new engine doesn't show wear and tear immediately. If we tell the model to look for degradation on Flight 1, it will get confused.
   - *The Solution:* I capped the Remaining Useful Life at **125 cycles**. The model learns that anything above 125 is simply "healthy," allowing it to dedicate its computational power to the actual drop-off curve near the end of the engine's life.
3. **Noise Reduction:** Conducted Exploratory Data Analysis (EDA) to find sensors (like Sensor 1 and 18) that remained completely constant throughout the engine's lifespan. These were dropped to improve model accuracy and speed.
4. **Model Training & Evaluation:** Trained a `RandomForestRegressor` on the cleaned data. The model operates as an ensemble of 100 decision trees, predicting the exact number of flights remaining. It is evaluated using **RMSE (Root Mean Squared Error)** to determine its real-world viability.

---

## 🛠️ Tech Stack

| Category | Technology/Library | Purpose |
| :--- | :--- | :--- |
| **Language** | `Python 3.8+` | Core programming language |
| **Data Manipulation** | `Pandas`, `NumPy` | Data wrangling, math operations, and dataframe management |
| **Machine Learning** | `Scikit-Learn` | Model building (Random Forest), preprocessing, and metrics |
| **Data Visualization** | `Matplotlib`, `Seaborn` | Plotting sensor degradation and model accuracy charts |
| **Environment** | `Jupyter Notebook` | Interactive development and step-by-step execution |

---

## 🚀 Getting Started

Follow these steps to run the pipeline on your local machine.

### Prerequisites
You will need Python installed on your system along with pip. 


