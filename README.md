# 🏥 UAC System Capacity & Care Load Analytics

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-Data%20Visualization-3F4F75?style=for-the-badge&logo=plotly)

🔗 **Live Dashboard:** [View Application Here](https://uac-capacity-intelligence.streamlit.app/)


## 📖 Overview

The UAC System Capacity & Care Load Analytics project is an enterprise-grade **Decision Intelligence Dashboard** designed to monitor, forecast, and optimize the Unaccompanied Alien Children (UAC) care pipeline. 

Moving beyond traditional descriptive analytics, this platform integrates **predictive modeling** and **scenario simulation** to empower proactive resource planning, bottleneck mitigation, and executive-level strategic interventions.

## ✨ Key Features

* **Automated Data Processing:** A robust preprocessing pipeline that handles messy, mixed date formats (`%b %d, %Y`, `MM/DD/YYYY`) from raw government CSVs and uses advanced **Time-Series Interpolation** to seamlessly fill NaNs without dropping critical data (handling up to 720+ continuous records).
* **Real-Time KPI Tracking & Semantic UI:** Instant calculation of critical metrics (Care Load Volatility, Net Intake Pressure) with UI cards that dynamically shift colors (Red/Green) based on threshold logic for instant glance-value.
* **Predictive Forecasting Engine:** Leverages `scikit-learn` Linear Regression and Rolling Averages to forecast System Load and Discharges over user-selected 7-day or 14-day horizons.
* **Scenario Simulation (What-If Analysis):** Interactive sliders allow policymakers to stress-test the system by adjusting discharge rates and intake volumes to visualize potential future systemic impacts.
* **Automated Insights Engine:** Evaluates live data trends and generates plain-text, prescriptive operational insights (e.g., flagging notable backlog streaks and suggesting procurement adjustments).

## ☑️ Core KPIs Monitored

1. **Total Children Under Care:** Combined real-time headcount across both CBP and HHS environments.
2. **Net Intake Pressure:** Calculates the daily net-new bottleneck (`Transfers Out` minus `Discharges`). *(Evaluated against threshold > 0)*
3. **Discharge Offset Ratio:** Measures HHS discharges relative to the HHS base population.
4. **Care Load Volatility Index:** A 14-day rolling standard deviation of the Total System Load to measure operational unpredictability.
5. **Backlog Accumulation Rate:** The percentage of days exhibiting positive intake pressure.

## 🛠️ Technology Stack

* **Frontend / UI:** Streamlit
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Visualizations:** Plotly Express / Graph Objects
* **Architecture:** Modular Python backend separating data loading, preprocessing, model fitting, and UI rendering.

## 📁 Project Structure

## 📁 Project Structure

uac_analytics/
│
├── app.py                  # Main Streamlit dashboard application
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── .gitignore              # Git configuration to ignore env and cache files
│
├── data/                   # Directory for raw data
│   └── HHS_Unaccompanied_Alien_Children_Program.csv 
│
├── reports/                # Directory for exported summaries and research
│   ├── executive_summary.md
│   └── research_paper.md
│
└── src/                    # Modular Python Backend
    ├── __init__.py
    ├── data_loader.py      # Handles CSV uploads and local file fetching
    ├── preprocessing.py    # Sanitizes inputs, handles dates, and interpolates NaNs
    ├── metrics.py          # Mathematical logic for KPI generation
    ├── forecasting.py      # ML predictive models (Scikit-Learn)
    ├── simulation.py       # Logic for scenario/what-if sliders
    └── insights.py         # Text generation for automated insights

## ⚙️ Installation & Setup

Clone the repository:
```bash
git clone https://github.com/vadubey78945-spec/UAC-Analytics-Dashboard.git
cd uac_analytics
```

Create a virtual environment (Recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

Install required dependencies:
```bash
pip install -r requirements.txt
```

Run the Dashboard:
```bash
streamlit run app.py
```

## 📊 Expected Data Format

The application is designed to intelligently parse datasets originating from HHS/CBP public releases. The core expected columns are:

* **Date** (Chronological record)
* **Children apprehended and placed in CBP custody*** (CBP Intake)
* **Children in CBP custody** (Current CBP load)
* **Children transferred out of CBP custody** (CBP to HHS transitions)
* **Children in HHS Care** (Current HHS load)
* **Children discharged from HHS Care** (HHS out-placements)

> ⚠️ **IMPORTANT NOTE FOR CUSTOM UPLOADS:** > If you are uploading a new custom CSV via the sidebar, **the dataset MUST contain the exact features (columns) listed above.** The robust preprocessing pipeline relies on these specific structural inputs. Uploading a dataset with missing or renamed columns will result in a pipeline validation error.

💡 Usage Guide

* Data Source: Use the sidebar to either upload a fresh CSV or use the bundled historical dataset.

* Forecasting: Select the forecast horizon (7 or 14 days) in the sidebar to project future system loads.

* Simulations: Navigate to the 'Forecast & Simulation' tab and adjust the sliders to run "What-If" scenarios on intake and discharge volumes.

⚖️ Disclaimer
This project was developed for academic/internship demonstration purposes. It utilizes publicly available or mocked structural data and should not be used as the sole basis for real-world policy decisions without official vetting.