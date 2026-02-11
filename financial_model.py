import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xlsxwriter

# --- CONFIGURATION & STYLING ---
# Professional Institutional Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
colors = {
    'nvidia_green': '#76B900',
    'dark_gray': '#333333',
    'slate_blue': '#4a69bd',
    'alert_red': '#e55039',
    'positive_green': '#009432'
}

class NvidiaInstitutionalModel:
    def __init__(self):
        # --- 1. HISTORICAL DATA (FY26 Consensus Estimates) ---
        # Updated from Audit: Yahoo Finance / Analyst Consensus
        self.current_revenue = 213.35 * 1e9  # $213.35B
        self.gross_margin_pct = 0.745        # 74.5% (Guidance/Consensus)
        self.op_margin_pct = 0.620           # 62.0% (Conservatism applied)
        
        # Derived Base Financials
        self.current_gross_profit = self.current_revenue * self.gross_margin_pct
        self.current_opex = self.current_revenue * (self.gross_margin_pct - self.op_margin_pct) # Back-solve OpEx
        self.current_op_income = self.current_revenue * self.op_margin_pct
        
        # Tax & Net Income
        self.tax_rate = 0.145
        self.current_net_income = self.current_op_income * (1 - self.tax_rate)
        
        # Capital Structure
        self.shares_outstanding = 24.6 * 1e9
        self.current_eps = self.current_net_income / self.shares_outstanding
        self.pe_multiple = 35.0  # Institutional Target Multiple
        self.current_share_price = self.current_eps * self.pe_multiple

        # --- 2. INVESTMENT PARAMETERS (The "Ask") ---
        self.new_headcount = 2000
        self.cost_per_head = 375000  # Increased to Institutional Standard (Stock-Based Comp incl.)
        self.investment_cost = self.new_headcount * self.cost_per_head  # ~$750M
        
        # --- 3. UNIT ECONOMICS (The "COGS Correction") ---
        # Institutional Standard: New revenue has a cost.
        # Mix assumption: 80% Hardware (70% GM) / 20% Software (90% GM) -> ~74% Blended GM
        self.incremental_cogs_pct = 0.26  # 26% of Revenue is COGS
        
        # --- 4. SCENARIOS (Stress Tested) ---
        self.scenarios = {
            "Cyclical Crash (-30%)": -0.30 * self.current_revenue, # Revenue COLLAPSE
            "Status Quo": 0,
            "Bear Case": 500 * 1e6,      # +$500M Revenue (Fail)
            "Base Case": 1.5 * 1e9,      # +$1.5B Revenue (2x Cost Coverage)
            "Bull Case (AI Demand)": 4.0 * 1e9, # +$4.0B Revenue
        }
        
        self.results = None

    def run_model(self):
        """Calculates financials with strict NOPAT/ROIC logic."""
        data = []
        
        for name, revenue_impact in self.scenarios.items():
            # 1. Pro Forma Revenue
            pf_revenue = self.current_revenue + revenue_impact
            
            # 2. Pro Forma COGS (Crucial Fix)
            # If revenue shrinks (Crash), COGS shrink too (Variable Cost). 
            # If revenue grows, we incur incremental COGS.
            if revenue_impact < 0:
                # Downside: Lost revenue saves COGS at standard rate
                incremental_cogs = revenue_impact * (1 - self.gross_margin_pct) # Saving COGS
            else:
                # Upside: New revenue costs money to make
                incremental_cogs = revenue_impact * self.incremental_cogs_pct
            
            pf_gross_profit = pf_revenue - ((self.current_revenue * (1-self.gross_margin_pct)) + incremental_cogs)
            
            # 3. Pro Forma OpEx (Fixed Cost Increases)
            # The $750M investment is FIXED cost. It happens regardless of revenue.
            pf_opex = self.current_opex + self.investment_cost
            
            # 4. Profitability
            pf_op_income = pf_gross_profit - pf_opex
            pf_net_income = pf_op_income * (1 - self.tax_rate)
            pf_eps = pf_net_income / self.shares_outstanding
            
            # 5. Incremental ROI Analysis (The "True" ROI)
            # NOPAT Impact = (New Sales - New COGS - New OpEx) * (1 - Tax)
            delta_op_income = pf_op_income - self.current_op_income
            delta_nopat = delta_op_income * (1 - self.tax_rate)
            
            # Return on Invested Capital (Incremental)
            # Numerator: Incremental NOPAT
            # Denominator: The $750M Investment
            roic = delta_nopat / self.investment_cost if self.investment_cost > 0 else 0
            
            # 6. Valuation Impact
            implied_share_price = pf_eps * self.pe_multiple
            value_creation = (implied_share_price - self.current_share_price) * self.shares_outstanding

            data.append({
                "Scenario": name,
                "Revenue ($B)": pf_revenue / 1e9,
                "Op Income ($B)": pf_op_income / 1e9,
                "Op Margin (%)": (pf_op_income / pf_revenue) * 100,
                "EPS ($)": pf_eps,
                "Implied Share Price ($)": implied_share_price,
                "Incr. NOPAT ($M)": delta_nopat / 1e6,
                "ROIC (%)": roic * 100,
                "Value Creation ($B)": value_creation / 1e9
            })
            
        self.results = pd.DataFrame(data)
        return self.results

    def generate_charts(self):
        """Generates institutional-grade visuals."""
        if self.results is None:
            self.run_model()
        
        df = self.results
        
        # --- Chart 1: ROIC vs Cost of Capital (Waterfall-ish Bar) ---
        plt.figure(figsize=(12, 7))
        
        # Color logic: Red if ROIC < WACC (10%), Green if > WACC
        wacc = 10.0
        clrs = [colors['alert_red'] if x < wacc else colors['positive_green'] for x in df['ROIC (%)']]
        
        ax = sns.barplot(x='Scenario', y='ROIC (%)', data=df, palette=clrs)
        plt.axhline(y=wacc, color='black', linestyle='--', linewidth=2, label='WACC Hurdle (10%)')
        plt.axhline(y=0, color='black', linewidth=1)
        
        plt.title('Incremental ROIC on $750M Investment', fontsize=18, fontweight='bold')
        plt.ylabel('Return on Invested Capital (%)')
        plt.legend(loc='upper left')
        
        # Annotate
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=12, fontweight='bold')
            
        plt.tight_layout()
        plt.savefig('visual_institutional_roic.png', dpi=300)
        plt.close()
        
        # --- Chart 2: Earnings Power Sensitivity (Double Y-Axis) ---
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        # Bar chart for Op Income
        sns.barplot(x='Scenario', y='Op Income ($B)', data=df, ax=ax1, color=colors['slate_blue'], alpha=0.6)
        ax1.set_ylabel('Operating Income ($B)', color=colors['slate_blue'], fontsize=14)
        ax1.tick_params(axis='y', labelcolor=colors['slate_blue'])
        
        # Line chart for Margin
        ax2 = ax1.twinx()
        sns.lineplot(x='Scenario', y='Op Margin (%)', data=df, ax=ax2, color=colors['dark_gray'], marker='o', linewidth=3)
        ax2.set_ylabel('Operating Margin (%)', color=colors['dark_gray'], fontsize=14)
        ax2.tick_params(axis='y', labelcolor=colors['dark_gray'])
        ax2.set_ylim(20, 80) # Force wide scale to show crash impact
        
        plt.title('Operating Leverage: Income vs. Margin', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig('visual_institutional_leverage.png', dpi=300)
        plt.close()

    def export_excel(self):
        """Exports an Institutional Model Audit Excel."""
        if self.results is None:
            self.run_model()
            
        file_name = 'Nvidia_Institutional_Model_v2.xlsx'
        writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
        workbook = writer.book
        
        # Formats
        header_fmt = workbook.add_format({'bold': True, 'font_color': 'white', 'bg_color': '#2c3e50', 'border': 1})
        num_fmt = workbook.add_format({'num_format': '#,##0.00'})
        pct_fmt = workbook.add_format({'num_format': '0.00%'})
        
        df = self.results
        df.to_excel(writer, sheet_name='Institutional_Model', index=False)
        ws = writer.sheets['Institutional_Model']
        
        # Apply Headers
        for idx, col in enumerate(df.columns):
            ws.write(0, idx, col, header_fmt)
            
        # Column Widths
        ws.set_column('A:A', 25) # Scenario Name
        ws.set_column('B:I', 15) # Data
        
        writer.close()
        print(f"✅ Institutional Report Generated: {file_name}")

if __name__ == "__main__":
    print("--- Running Institutional Grade Audit Model ---")
    model = NvidiaInstitutionalModel()
    model.run_model()
    model.generate_charts()
    model.export_excel()
    print("--- Audit Complete ---")
