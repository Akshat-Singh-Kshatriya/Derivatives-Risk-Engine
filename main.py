from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from apscheduler.schedulers.background import BackgroundScheduler
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from arch import arch_model
import warnings

warnings.filterwarnings('ignore')

app = FastAPI(title="Options Pricing API")

# --- 1. THE 9 AM RBI SCRAPER ---
def fetch_rbi_rate():
    global LATEST_RISK_FREE_RATE
    url = "https://tradingeconomics.com/india/government-bond-yield"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            val_element = soup.select_one("#ctl00_ContentPlaceHolder1_ctl00_ctl01_Panel1 tr:nth-of-type(2) td:nth-of-type(2)")
            if val_element:
                LATEST_RISK_FREE_RATE = float(val_element.get_text().strip())
    except Exception as e:
        pass

scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
scheduler.add_job(fetch_rbi_rate, 'cron', hour=9, minute=0)
scheduler.start()

# --- 2. QUANTITATIVE MODELS ---
class VolatilityForecaster:
    @staticmethod
    def calc_ewma(returns, lambda_param=0.94):
        variances = np.zeros_like(returns)
        variances[0] = np.var(returns)
        for i in range(1, len(returns)):
            variances[i] = lambda_param * variances[i-1] + (1 - lambda_param) * returns.iloc[i-1]**2
        return np.sqrt(variances[-1]) * np.sqrt(252)

    @staticmethod
    def calc_garch(returns):
        try:
            scaled_returns = returns.dropna() * 100 
            am = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='normal')
            res = am.fit(disp='off')
            next_day_var = res.forecast(horizon=1).variance.iloc[-1, 0]
            return np.sqrt(next_day_var / 10000) * np.sqrt(252)
        except:
            return returns.std() * np.sqrt(252)

class DerivativesEngine:
    def __init__(self, S, K, T, r, sigma):
        self.S = S; self.K = K; self.T = T; self.r = r; self.sigma = sigma

    def get_prices(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        bs_call = self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)

        steps = 50
        dt = self.T / steps
        u = np.exp(self.sigma * np.sqrt(dt)); d = 1 / u
        p = (np.exp(self.r * dt) - d) / (u - d)
        asset_prices = self.S * (u ** np.arange(steps, -1, -1)) * (d ** np.arange(0, steps + 1))
        values = np.maximum(0, asset_prices - self.K)
        for i in range(steps - 1, -1, -1):
            continuation = np.exp(-self.r * dt) * (p * values[:-1] + (1 - p) * values[1:])
            asset_prices = self.S * (u ** np.arange(i, -1, -1)) * (d ** np.arange(0, i + 1))
            intrinsic = np.maximum(0, asset_prices - self.K)
            values = np.maximum(continuation, intrinsic)
        am_call = values[0]

        return bs_call, am_call

    def get_greeks(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
        vega = self.S * np.sqrt(self.T) * norm.pdf(d1) / 100
        theta = - (self.S * self.sigma * norm.pdf(d1)) / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d1 - self.sigma * np.sqrt(self.T))
        return {'Delta': round(delta, 3), 'Gamma': round(gamma, 4), 'Vega': round(vega, 2), 'Theta': round(theta/365, 2)}

# --- 3. API ENDPOINTS ---
@app.get("/api/analyze")
def analyze_stock(ticker: str = "DIVISLAB.NS", strike_pct: float = 100.0, days_expiry: int = 30, vol_model: str = "garch"):
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period="1y")
        if history.empty: return {"status": "error", "message": "Invalid Ticker or No Data"}
        
        S0 = float(history['Close'].iloc[-1])
        history['Log_Ret'] = np.log(history['Close'] / history['Close'].shift(1))
        returns = history['Log_Ret'].dropna()

        if vol_model == "garch": sigma = VolatilityForecaster.calc_garch(returns)
        elif vol_model == "ewma": sigma = VolatilityForecaster.calc_ewma(returns)
        else: sigma = returns.tail(60).std() * np.sqrt(252)

        Strike = round(S0 * (strike_pct / 100) / 50) * 50
        T = days_expiry / 365
        r_decimal = LATEST_RISK_FREE_RATE / 100.0

        engine = DerivativesEngine(S0, Strike, T, r_decimal, sigma)
        bs_price, am_price = engine.get_prices()
        greeks = engine.get_greeks()
        
        # New: Intrinsic vs Time Value Breakdown
        intrinsic_val = max(0, S0 - Strike)
        time_val = max(0, bs_price - intrinsic_val)

        dt = T / 100
        Z = np.random.standard_normal((50, 100))
        paths = np.hstack([np.full((50, 1), S0), S0 * np.exp(np.cumsum((r_decimal - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1))])
        mc_price = np.exp(-r_decimal * T) * np.mean(np.maximum(np.mean(paths[:, 1:], axis=1) - Strike, 0))

        exposure = S0 * 100 
        var_param = exposure * (sigma / np.sqrt(252)) * 2.326
        var_hist = abs(exposure * np.percentile(history['Close'].pct_change().dropna(), 1))
        
        strikes_plot = np.linspace(S0*0.8, S0*1.2, 20)
        smile_vols = sigma + 0.5 * ((strikes_plot - S0)/S0)**2

        return {
            "status": "success",
            "market": {
                "ticker": ticker, "spot": round(S0, 2), "strike": Strike, 
                "volatility": round(sigma * 100, 2), "risk_free_rate": LATEST_RISK_FREE_RATE
            },
            "pricing": {
                "bsm_european": round(bs_price, 2), "binomial_american": round(am_price, 2), "asian_mc": round(mc_price, 2),
                "intrinsic_value": round(intrinsic_val, 2), "time_value": round(time_val, 2)
            },
            "greeks": greeks,
            "risk": {"var_param": round(var_param, 2), "var_hist": round(var_hist, 2), "exposure": round(exposure, 2)},
            "charts": {
                "mc_paths": paths.tolist(), "smile_x": strikes_plot.tolist(), "smile_y": (smile_vols * 100).tolist()
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    with open("index.html", "r") as f: return f.read()

fetch_rbi_rate()
