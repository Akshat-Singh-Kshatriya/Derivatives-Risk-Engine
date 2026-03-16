# Derivatives-Risk-Engine

It is a high-performance, real-time quantitative finance engine designed for options pricing, volatility forecasting, and portfolio risk analysis. 

## 🚀 Key Achievements & Features

### 1. Advanced Volatility Forecasting
Standard historical standard deviation is insufficient for live trading. This engine implements advanced econometric models to forecast future variance:
* **GARCH(1,1):** Generalized Autoregressive Conditional Heteroskedasticity for dynamic, mean-reverting volatility forecasting.
* **EWMA (RiskMetrics):** Exponentially Weighted Moving Average (with λ = 0.94) to heavily weight recent market shocks.
* **Historical (60-Day):** Baseline standard deviation annualized for comparison.

### 2. Multi-Model Derivatives Pricing
Calculates fair value across different exercise styles and path dependencies:
* **European Options:** Black-Scholes-Merton (BSM) closed-form solutions.
* **American Options:** 50-step Binomial Tree modeling for early exercise premiums.
* **Asian/Exotic Options:** Monte Carlo simulations (Geometric Brownian Motion) across 50 simulated price paths to calculate average-price payoffs.
* **Intrinsic vs. Time Value:** Dynamic breakdown of the option's premium.

### 3. Risk Management & Greeks
* **First & Second Order Greeks:** Full calculation of Delta (Δ), Gamma (Γ), Theta (Θ), and Vega (ν) using BSM partial derivatives.
* **Value at Risk (VaR):** Calculates both **Parametric VaR (99%)** using $Z$-scores and **Historical VaR (99%)** using actual percentile drops over the trailing year to determine the maximum expected loss on a 1-lot exposure.

### 4. Resilient Architecture & Macro Data
* **Cloudflare Bypass:** Directly scraping Indian 10-Year Bond yields from sites like Trading Economics results in immediate IP bans from cloud providers like Railway. This engine programmatically fetches the live US 10-Year Treasury Yield (`^TNX`) via API and applies the historical US-India Sovereign Spread (~2.50%) to generate a dynamically updating, unblockable Indian Risk-Free Rate.
* **Streamlit-Style UI, Zero Overhead:** Recreated the clean, metric-driven layout of Streamlit using raw HTML/CSS and Plotly.js, resulting in blazing fast load times and perfect SEO compatibility.

## 🛠️ Tech Stack

* **Backend:** Python 3, FastAPI, Uvicorn
* **Quant/Math:** NumPy, Pandas, SciPy, `arch` (for GARCH models)
* **Data Ingestion:** `yfinance`
* **Frontend:** HTML5, CSS3, Vanilla JavaScript, Plotly.js
* **Deployment:** Railway.app 

## 💻 Running Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Akshat-Singh-Kshatriya/Derivatives-Risk-Engine
   cd Derivatives-Risk-Engine
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
3. **Start the FastAPI server**
   ```bash
   uvicorn main:app --reload
