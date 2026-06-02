# 🚢 Titanic: Data Analysis & Predictive Insights

![Python](https://img.shields.io/badge/Python-3.8+-blue)

![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)

![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-orange)

An end-to-end exploratory data analysis (EDA) pipeline that processes the historical Titanic passenger manifest, engineers robust features, and extracts actionable demographic insights. This project simulates a production-ready analytical workflow.

---

## 📌 Executive Summary
This project analyzes the factors affecting survival rates during the Titanic disaster. By cleaning noisy historical data and leveraging statistical visualization techniques, this pipeline uncovers critical social, economic, and demographic drivers behind passenger survival.

### Key Insights Discovered:
* **The "Women and Children First" Protocol:** Verified mathematically via demographic survival gaps.
* **Socio-Economic Leverage:** Clear positive correlation between ticket pricing (`Fare`), upper-class placement (`Pclass`), and survival outcomes.
* **Age Demographics:** Skewed distributions highlight critical survival clusters among specific age brackets.

---

## 🛠️ Tech Stack & Architecture
* **Language:** Python 3.8+
* **Data Wrangling:** `pandas`, `numpy`
* **Data Visualization:** `seaborn`, `matplotlib`
* **Environment Management:** `os`, `pathlib`

---

## 📂 Project Structure
```text
Data-Analysis/
│
├── data/
│   └── titanic.csv                # Raw passenger manifest dataset
│
├── visualizations/                # Production-grade exploratory plots
│   ├── bar_chart.png              # Survival Rate vs Passenger Class
│   ├── scatter_plot.png           # Bi-variate analysis of Age vs Fare 
│   ├── heatmap.png                # Multi-variate Pearson Correlation matrix
│   ├── age_histogram.png          # Univariate Age distribution density
│   └── gender_survival.png        # Gender-based survival comparison
│
├── insights.txt                   # Auto-generated textual analytical summary
├── summary.csv                    # Quantified high-level KPIs matrix
├── analysis.py                    # Modular Core Execution Script
├── requirements.txt               # Project dependencies
└── README.md                      # Documentation
```

---

## ⚙️ Data Pipeline Workflow

### 1. Robust Data Ingestion & Sanitization
* **Dynamic Loading:** Protected file-reading mechanism with localized exception handling.
* **Missing Value Imputation:** 
  * `Age` fields are filled using a stable **median** strategy to minimize outlier distortion.
  * `Embarked` location gaps are imputed using the mathematical **mode**.
* **Dimensionality Reduction:** Dropped high-cardinality features (`Cabin`) to preserve model/statistical hygiene.

### 2. Feature Encoding
* Transformed categorical variables (`Sex`) into numerical vectors (`Sex_encoded`) to enable Pearson correlation modeling.

### 3. Automated Output Generation
* Generates an automated, shareable business summary (`summary.csv`).
* Writes key bulleted metrics to an executive brief (`insights.txt`).

---

## 📊 Sample Insights & Visualizations Preview

The automated analytical script yields the following statistical correlations:
* **Socio-Economic Status:** First-class passengers achieved structural survival priority compared to lower decks.
* **Gender Vector:** Disproportionately higher survival velocities observed within female subsets due to maritime boarding rules.

*(All generated publication-quality figures are stored automatically inside the `/visualizations` directory).*

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed on your system. 

### Installation & Execution
1. **Clone the repository:**
   ```bash
   git clone https://github.com/shubham88c-pr/data-analysis.git
   cd data_analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analysis pipeline:**
   ```bash
   python analysis.py
   ```

---

## 📈 Future Scope
* Implement advanced machine learning models (Random Forest / XGBoost) to predict survival probability.
* Deploy an interactive web dashboard using **Streamlit** for real-time user-driven filtering.

---
💡 *Developed as a part of a professional Portfolio Showcase. For inquiries or collaborations, feel free to reach out via GitHub.*
