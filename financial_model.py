import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xlsxwriter
from scipy.stats import norm

# --- CONFIGURATION & STYLING ---
# Buy-Side / Investment Committee Style
plt.style.use('dark_background')
sns.set_context("talk")
colors = {
    'nvidia_green': '#76B900',
    'bear_red': '#ff4d4d',
    'bull_blue': '#4a69bd',
    'neutral_gray': '#95a5a6',
    'implied_orange': '#e67e22'
}

class BuySideStressTest:
    def __init__(self):
        # --- 1. CURRENT MARKET SNAPSHOT ---
        self.share_price = 140.00
        self.shares_outstanding = 24.6 * 1e9
        self.market_cap = self.share_price * self.shares_outstanding # ~$3.44T
        
        # --- 2. BASELINE FUNDAMENTALS (Consensus FY26) ---
        self.revenue_base = 213.35 * 1e9
        self.fcf_margin = 0.45 # Free Cash Flow Margin (High, but CAPEX heavy)
        self.wacc = 0.095 # 9.5% Cost of Capital
        self.terminal_growth = 0.03 # 3% GDP+
        
        # --- 3. SCENARIO INPUTS (The "Kill" Variables) ---
        self.scenarios = {
            "Consensus (Priced In)": {
                "Rev_CAGR_5y": 0.20, "Term_Margin": 0.55, "Market_Share": 0.90
            },
            "Competitive Erosion (AMD/TPU)": {
                "Rev_CAGR_5y": 0.12, "Term_Margin": 0.40, "Market_Share": 0.65
            },
            "Cycle Crash (History Repeats)": {
                "Rev_CAGR_5y": 0.05, "Term_Margin": 0.35, "Market_Share": 0.70
            },
            "AI Supercycle (The Bull)": {
                "Rev_CAGR_5y": 0.30, "Term_Margin": 0.60, "Market_Share": 0.95
            }
        }
        
    def reverse_dcf(self):
        """Reverse Engineers what growth is priced into the stock."""
        # Simple single-stage reverse solve for implied growth
        # Value = FCF / (WACC - g) -> implied g = WACC - (FCF / Value)
        current_fcf = self.revenue_base * self.fcf_margin
        implied_perpetual_growth = self.wacc - (current_fcf / self.market_cap)
        
        # This is a crude approximation for "Market Implied Expectations"
        # Converting perpetuity g to a 5y CAGR equivalent roughly:
        implied_5y_cagr = implied_perpetual_growth * 5.0 # Heuristic multiplier
        
        return {
            "Implied_Perpetual_Growth": implied_perpetual_growth,
            "Implied_Revenue_CAGR": max(0.25, implied_5y_cagr) # Floor at 25% because market pays for frontend growth
        }

    def run_valuation_scenarios(self):
        """Runs 5-year DCF for each scenario."""
        results = []
        
        for name, params in self.scenarios.items():
            # Project 5 years
            rev = [self.revenue_base * ((1 + params["Rev_CAGR_5y"]) ** i) for i in range(1, 6)]
            fcf = [r * params["Term_Margin"] for r in rev]
            
            # Terminal Value
            tv = (fcf[-1] * (1 + self.terminal_growth)) / (self.wacc - self.terminal_growth)
            
            # Discount Back
            dcf_val = 0
            for i, cash in enumerate(fcf):
                dcf_val += cash / ((1 + self.wacc) ** (i + 1))
            
            dcf_val += tv / ((1 + self.wacc) ** 5)
            
            implied_share_price = dcf_val / self.shares_outstanding
            upside = (implied_share_price - self.share_price) / self.share_price
            
            results.append({
                "Scenario": name,
                "Implied Price ($)": implied_share_price,
                "Upside/Downside (%)": upside * 100,
                "5y Revenue ($B)": rev[-1] / 1e9,
                "Assumed CAGR": params["Rev_CAGR_5y"]
            })
            
        return pd.DataFrame(results)

    def generate_charts(self):
        """Generates Buy-Side Visuals."""
        df = self.run_valuation_scenarios()
        
        # --- Chart 1: Market Implied Expectations Bridge ---
        # Comparing Model scenarios to Current Price
        plt.figure(figsize=(12, 6))
        
        # Sort for visual flow
        df = df.sort_values("Implied Price ($)")
        
        colors_list = [colors['bear_red'] if x < 0 else colors['bull_blue'] for x in df['Upside/Downside (%)']]
        
        ax = sns.barplot(x='Scenario', y='Implied Price ($)', data=df, palette=colors_list)
        plt.axhline(y=self.share_price, color=colors['implied_orange'], linestyle='--', linewidth=3, label=f'Current Price (${self.share_price})')
        
        plt.title('Valuation Reality Check: What is Priced In?', fontsize=18, fontweight='bold')
        plt.ylabel('Implied Share Price ($)')
        plt.legend()
        
        for container in ax.containers:
            ax.bar_label(container, fmt='$%.0f', padding=3, fontsize=12, fontweight='bold')
            
        plt.tight_layout()
        plt.savefig('visual_buyside_valuation.png')
        plt.close()
        
        # --- Chart 2: Risk Distribution (Bell Curve) ---
        # Visualizing the probability of outcomes
        plt.figure(figsize=(10, 6))
        
        # Generate a distribution based on our Base Case w/ std dev
        mu = df.loc[df['Scenario'] == "Consensus (Priced In)", "Implied Price ($)"].values[0]
        sigma = 40 # High volatility assumption
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        plt.plot(x, norm.pdf(x, mu, sigma), color='white', linewidth=2)
        
        # Fill areas
        plt.fill_between(x, norm.pdf(x, mu, sigma), where=(x < self.share_price), color=colors['bear_red'], alpha=0.3, label='Downside Risk')
        plt.fill_between(x, norm.pdf(x, mu, sigma), where=(x >= self.share_price), color=colors['nvidia_green'], alpha=0.3, label='Upside Potential')
        
        plt.axvline(self.share_price, color=colors['implied_orange'], linestyle=':', label='Current Price')
        
        plt.title('Probability Distribution of Valuation Outcomes', fontsize=16, fontweight='bold')
        plt.xlabel('Share Price ($)')
        plt.ylabel('Probability Density')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('visual_buyside_risk.png')
        plt.close()

if __name__ == "__main__":
    print("--- Running Buy-Side Stress Test ---")
    model = BuySideStressTest()
    df = model.run_valuation_scenarios()
    print(df)
    model.generate_charts()
    print("--- Analysis Complete ---")
