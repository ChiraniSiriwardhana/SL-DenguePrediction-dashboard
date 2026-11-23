# 🦟 Sri Lanka Dengue Risk Prediction

Weekly, district-level dengue forecasting for Sri Lanka, with a comparison of ensemble learning and hybrid models.

[![Live dashboard](https://img.shields.io/badge/Streamlit-live%20dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://dengue-prediction-ensemble.streamlit.app/)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BkcAJhWFWFsqvuumIJKepReC2PNs3FQH)

| | Link |
|---|---|
| 📊 Live dashboard | https://dengue-prediction-ensemble.streamlit.app/ |
| 📓 Training notebook (Google Colab) | https://colab.research.google.com/drive/1BkcAJhWFWFsqvuumIJKepReC2PNs3FQH |
| 🗂️ Dataset in this repository | [`data/Sri_Lanka_Dengue_Dataset.csv`](data/Sri_Lanka_Dengue_Dataset.csv) |

(Free Streamlit apps go to sleep when nobody uses them. If you see a "wake up" button, press it and wait a minute.)

> ⚠️ **Disclaimer.** This is a university research project. Its estimates are **not** official health warnings and do
> not confirm an outbreak. For official information, contact the Ministry of Health or the National Dengue Control Unit.

---

## What the app is

![Dashboard  1](images/dashboard1.png)

![Dashboard  2](images/dashboard2.png)

Using weekly dengue cases and weather data for the 25 districts of Sri Lanka (2007 to July 2024), the project predicts
the number of reported dengue cases in the **next week**. It has two parts:

- **A notebook** (`Sri_Lanka_Dengue_Ensemble_Learning.ipynb`) that prepares the data, trains the models and evaluates
  them. Models compared: Naive, Linear Regression, ARIMA, Random Forest (bagging), XGBoost, LightGBM and CatBoost
  (boosting), Stacking, and a Hybrid ARIMA + Random Forest.
- **A Streamlit dashboard** (`app.py`) that shows the results and lets users try a what-if prediction.


## Folder structure

```
SL-DenguePrediction-dashboard/
├── app.py                                    # the Streamlit dashboard
├── requirements.txt                          # Python packages of the dashboard
├── Sri_Lanka_Dengue_Ensemble_Learning.ipynb  # data preparation, training and evaluation (run it in Colab)
├── .streamlit/
│   └── config.toml                           # dark theme of the dashboard
├── data/
│   └── Sri_Lanka_Dengue_Dataset.csv          # the raw dataset (used by the notebook and by the dashboard)
├── sri_lanka_districts.geojson               # district boundaries for the maps (Natural Earth, simplified)
│
│   # Result files created by the notebook (Section 15)
├── predictions.csv                           # every model's prediction for the validation and test weeks
├── model_comparison.csv                      # MAE, RMSE and R² of every model
├── risk_thresholds.csv                       # district median and 80th percentile used for the risk levels
├── forecast_next_week.csv                    # forecast of every model for the latest week
├── training_time.csv                         # training time of every model
│
│   # Files of the Dengue Predictor page, created by the notebook's export cells
├── rf_model.joblib                           # trained Random Forest
├── catboost_model.cbm                        # trained CatBoost (it has a stacking weight above zero)
├── predictor_params.json                     # Linear Regression coefficients, stacking weights, ARIMA parameters
├── district_week_profile.csv                 # typical cases and weather for every district and week of the year
└── whatif_backtest.csv                       # accuracy of the Dengue Predictor shortcut on the test weeks
```

The dashboard looks for each file next to `app.py` or inside the `data/` folder.

---
## Run the notebook

### Recommended: Google Colab

1. Open the notebook with the Colab link: https://colab.research.google.com/drive/1BkcAJhWFWFsqvuumIJKepReC2PNs3FQH
2. (Optional) **File → Save a copy in Drive**, if you want to change anything.
3. Choose **Runtime → Run all** and wait. The notebook installs its libraries, loads the data from GitHub, trains all
   models and evaluates them. It takes several minutes; the Random Forest tuning and the stacking folds are the
   slowest parts.

> **Note for reviewers:** This notebook is written for Google Colab. Please open it through the Colab link rather than
> downloading it and running it in a local Jupyter environment. The dataset is loaded automatically from the project's
> public GitHub repository, so no file upload or setup is needed. Simply open the link and select
> **Runtime → Run all**.

### If you download the notebook and run it locally

If you download the notebook (for example from GitHub) and run it locally, it will still work, but please note:

- An **internet connection is required** (the data is loaded from GitHub).
- The required libraries must be installed (see [Requirements](#requirements)).
- The Colab-only download cells in Section 15 are skipped, and the output CSV files are saved in the notebook's
  working folder instead.
  
---

## The dataset and how to use it

The raw dataset is stored in this repository at **`data/Sri_Lanka_Dengue_Dataset.csv`**. Please keep this exact name and
folder, because both the notebook and the dashboard read it from there.

- **Source:** [rajaththomson/sri-lanka-dengue-fever-analysis](https://github.com/rajaththomson/sri-lanka-dengue-fever-analysis).
  Please check the original repository for its terms of use.

**Use it in your own code**

```python
import pandas as pd

# 1) from GitHub (works anywhere with internet; this is what the notebook does)
URL = "https://raw.githubusercontent.com/ChiraniSiriwardhana/SL-DenguePrediction-dashboard/refs/heads/main/data/Sri_Lanka_Dengue_Dataset.csv"
df = pd.read_csv(URL)

# 2) from a clone of the repository
df = pd.read_csv("data/Sri_Lanka_Dengue_Dataset.csv")

df["Week_Start_Date"] = pd.to_datetime(df["Week_Start_Date"], format="%m/%d/%Y")
df = df.sort_values(["District", "Week_Start_Date"]).reset_index(drop=True)
```

- **Notebook:** Section 3 loads the data from the GitHub link above (`DATA_URL`), so no upload is needed.
  If you move or rename the file, or make the repository private, change `DATA_URL`.
- **Dashboard:** `app.py` reads `data/Sri_Lanka_Dengue_Dataset.csv` for the Dataset page and for the Dengue Predictor.

---

## Run the dashboard on your computer (clone)

You need Python 3.10 or newer and Git.

```bash
git clone https://github.com/ChiraniSiriwardhana/SL-DenguePrediction-dashboard.git
cd SL-DenguePrediction-dashboard

python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at http://localhost:8501. You do **not** need to train anything: the result files and the trained
models are already in the repository.

**Troubleshooting**

- *A warning or error when loading `rf_model.joblib`:* the model was saved with a specific scikit-learn version.
  Install exactly that version (the one printed by the notebook's export cell), for example
  `pip install scikit-learn==<version>`, and write the same version in `requirements.txt`.
- *"File not found":* check that the files listed in [Folder structure](#folder-structure) are next to `app.py`
  (or inside `data/`).
- *The Dengue Predictor shows an error:* it needs `rf_model.joblib`, `predictor_params.json`,
  `district_week_profile.csv`, `whatif_backtest.csv` and the model files of the stacking members
  (for example `catboost_model.cbm`).

---

## Requirements

### Dashboard (`requirements.txt`)

```
streamlit>=1.50
pandas
numpy
plotly>=5.24
joblib
scikit-learn
catboost
```

---

## Recreate the result and model files

1. Run the whole notebook (Runtime → Run all, or **Run → Run All Cells** in Jupyter).
2. Run the export cells at the end of the notebook. They create the result files and the Dengue Predictor files listed
   in [Folder structure](#folder-structure), and print checks that the saved models give the same numbers as the models
   in the notebook.
3. Copy the files into the repository (same names, next to `app.py`) and start the dashboard again.

---

## Deploy your own copy

1. Fork this repository.
2. On [share.streamlit.io](https://share.streamlit.io) choose **Create app → Deploy a public app from GitHub**.
3. Select your fork, branch `main`, main file `app.py`, then **Deploy**.

---

### Method in brief

- **Cleaning:** duplicate rainfall column and sunrise/sunset columns removed (19 → 16 columns).
- **Target:** reported cases in the next week.
- **Features (24):** current cases, case lags (1, 2, 4, 8, 12 weeks) and a 4-week average; current weather; rainfall lags
  (2–10 weeks) and totals (4, 8, 12 weeks); maximum-temperature lags (4, 8 weeks); week of the year (sin, cos);
  district. All lag features use only past information.
- **Split (chronological, never random):** training 2007-03-26 to 2020-12-28, validation 2021–2022,
  test 2023-01-02 to 2024-07-08. Hyperparameters were chosen on the validation set. The weight-constrained stacking was
  added after the first (Ridge) stacking performed poorly on the test set; both are reported in the notebook.
- **Metrics:** MAE, RMSE, R². MAPE is shown only on weeks with at least one case, because about 13% of the weeks have
  zero cases.

### Limitations

- The data end in July 2024. This is a demonstration, not a live surveillance system.
- The **Dengue Predictor** fills 16 of the 24 model features with district-week averages. It is a what-if tool, not a
  validated forecast; its accuracy was measured with a backtest shown on that page, and the numbers you enter carry
  almost the whole prediction.
- The models predict *reported* cases, not true infections, and large outbreaks are hard to predict. Tree-based models
  cannot predict beyond the case numbers they have seen.
- The test period has more cases than the training period (distribution shift), so test errors are larger.
- Reporting delays would reduce accuracy in real use.
- Nothing on this dashboard is an official health warning.

---

## Acknowledgements

- Data: [rajaththomson/sri-lanka-dengue-fever-analysis](https://github.com/rajaththomson/sri-lanka-dengue-fever-analysis).
- District boundaries: [Natural Earth](https://www.naturalearthdata.com/) (public domain), simplified for this dashboard.
- Tools: Google Colab, scikit-learn, XGBoost, LightGBM, CatBoost, statsmodels, Streamlit, Plotly.
