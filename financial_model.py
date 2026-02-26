import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xlsxwriter

# --- CONFIGURATION & STYLING ---
# Set a professional style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("talk")
colors = {
    'nvidia_green': '#76B900',
    'dark_gray': '#333333',
    'light_gray': '#E0E0E0',
    'alert_red': '#D32F2F',
    'neutral_blue': '#1976D2'
}

class NvidiaModel:
    def __init__(self):
        # --- 1. HISTORICAL DATA (Est. FY2025/26 TTM) ---
        # Hardware/Software/Services Revenue
        self.current_revenue = 120.0 * 1e9  # $120B
        # Cost of Revenue + OpEx
        self.current_opex = 45.0 * 1e9      # $45B
        # Operating Income
        self.current_op_income = self.current_revenue - self.current_opex
        # Net Income (Approx 15% Tax Rate assumption for simplicity on Op Income)
        self.tax_rate = 0.145
        self.current_net_income = self.current_op_income * (1 - self.tax_rate)
        
        # Shares Outstanding (Diluted)
        self.shares_outstanding = 24.6 * 1e9
        
        # Headcount Estimates
        self.current_headcount = 32000  # Approx.
        
        # --- 2. INVESTMENT PARAMETERS ---
        self.new_headcount = 2000
        self.cost_per_head = 350000
        self.investment_cost = self.new_headcount * self.cost_per_head  # $700M
        
        # --- 3. SCENARIOS ---
        # Dictionary of scenarios: Market Conditions -> Impact on Revenue
        # We assume the investment is made, but the RETURN depends on the market.
        self.scenarios = {
            "Status Quo": 0,
            "Bear Case": 500 * 1e6,      # +$500M (Not enough to cover cost)
            "Base Case": 1.2 * 1e9,      # +$1.2B (Profitable)
            "Bull Case": 3.5 * 1e9,      # +$3.5B (High ROI)
            "AI Supercycle": 8.0 * 1e9   # +$8.0B (Massive Success)
        }
        
        self.results = None

    def run_model(self):
        """Calculates financials for each scenario."""
        data = []
        
        for name, new_sales in self.scenarios.items():
            # Pro Forma Financials
            pf_revenue = self.current_revenue + new_sales
            pf_opex = self.current_opex + self.investment_cost
            pf_op_income = pf_revenue - pf_opex
            pf_net_income = pf_op_income * (1 - self.tax_rate)
            
            # Margins
            pf_op_margin = pf_op_income / pf_revenue
            
            # EPS
            pf_eps = pf_net_income / self.shares_outstanding
            current_eps = self.current_net_income / self.shares_outstanding
            eps_accretion = (pf_eps - current_eps) / current_eps
            
            # Efficiency Metrics
            pf_headcount = self.current_headcount + self.new_headcount
            rev_per_employee = pf_revenue / pf_headcount
            
            # ROI
            roi = (new_sales - self.investment_cost) / self.investment_cost if self.investment_cost > 0 else 0

            data.append({
                "Scenario": name,
                "Revenue ($B)": pf_revenue / 1e9,
                "OpEx ($B)": pf_opex / 1e9,
                "Op Income ($B)": pf_op_income / 1e9,
                "Net Income ($B)": pf_net_income / 1e9,
                "EPS ($)": pf_eps,
                "EPS Accretion (%)": eps_accretion * 100,
                "Op Margin (%)": pf_op_margin * 100,
                "Rev/Employee ($M)": rev_per_employee / 1e6,
                "ROI (%)": roi * 100,
                "Investment Cost ($M)": self.investment_cost / 1e6,
                "New Sales ($B)": new_sales / 1e9
            })
            
        self.results = pd.DataFrame(data)
        return self.results

    def generate_charts(self):
        """Generates professional visualizations."""
        if self.results is None:
            self.run_model()
            
        df = self.results
        
        # --- Chart 1: EPS Impact (Bar Chart) ---
        plt.figure(figsize=(12, 7))
        ax = sns.barplot(x='Scenario', y='EPS ($)', data=df, palette="viridis")
        
        # Add baseline line
        current_eps = self.current_net_income / self.shares_outstanding
        plt.axhline(y=current_eps, color=colors['alert_red'], linestyle='--', linewidth=2, label=f'Current EPS (${current_eps:.2f})')
        
        plt.title('Projected EPS Impact per Scenario', fontsize=18, fontweight='bold', pad=20)
        plt.ylabel('Earnings Per Share ($)', fontsize=14)
        plt.xlabel('')
        plt.ylim(bottom=current_eps * 0.95, top=df['EPS ($)'].max() * 1.05)
        plt.legend()
        
        # Annotate bars
        for container in ax.containers:
            ax.bar_label(container, fmt='$%.3f', padding=3, fontsize=12, fontweight='bold')
            
        plt.tight_layout()
        plt.savefig('visual_eps_impact.png', dpi=300)
        print("✅ Chart Generated: visual_eps_impact.png")
        plt.close()

        # --- Chart 2: Efficiency (Revenue per Employee) ---
        plt.figure(figsize=(10, 6))
        # Compare Current vs Scenarios
        # Create a temp DF for plotting
        current_rpe = self.current_revenue / self.current_headcount / 1e6
        
        # Plot
        ax = sns.lineplot(x='Scenario', y='Rev/Employee ($M)', data=df, marker='o', linewidth=3, color=colors['nvidia_green'])
        plt.axhline(y=current_rpe, color='gray', linestyle=':', label=f'Current Efficiency (${current_rpe:.2f}M)')
        
        plt.title('Sales Efficiency: Revenue per Employee', fontsize=16, fontweight='bold')
        plt.ylabel('Revenue per Employee ($M)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('visual_efficiency.png', dpi=300)
        print("✅ Chart Generated: visual_efficiency.png")
        plt.close()

        # --- Chart 3: Sensitivity Heatmap (Cost vs Revenue) ---
        # Generate data for heatmap
        costs = np.linspace(500, 1000, 6) # $500M to $1B
        revenues = np.linspace(0, 2000, 6) # $0 to $2B
        
        heatmap_data = np.zeros((len(costs), len(revenues)))
        
        for i, c in enumerate(costs):
            for j, r in enumerate(revenues):
                # Calculate ROI
                cost_val = c * 1e6
                rev_val = r * 1e6
                roi = (rev_val - cost_val) / cost_val
                heatmap_data[i, j] = roi * 100

        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="RdYlGn", 
                         xticklabels=[f"${x/1000:.1f}B" for x in revenues],
                         yticklabels=[f"${y:.0f}M" for y in costs],
                         cbar_kws={'label': 'ROI (%)'})
        
        plt.title('Investment ROI Sensitivity\n(Revenue vs. Headcount Cost)', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('New Revenue Generated')
        plt.ylabel('Total Investment Cost')
        
        plt.tight_layout()
        plt.savefig('visual_sensitivity_heatmap.png', dpi=300)
        print("✅ Chart Generated: visual_sensitivity_heatmap.png")
        plt.close()

    def export_excel(self):
        """Exports a formatted Excel report."""
        if self.results is None:
            self.run_model()

        file_name = 'Nvidia_Financial_Expansion_Model.xlsx'
        writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
        workbook = writer.book
        
        # Formats
        fmt_header = workbook.add_format({'bold': True, 'font_color': 'white', 'bg_color': '#76B900', 'border': 1})
        fmt_currency = workbook.add_format({'num_format': '$#,##0.00'})
        fmt_percent = workbook.add_format({'num_format': '0.00%'})
        fmt_bold = workbook.add_format({'bold': True})
        
        # Sheet 1: Dashboard
        df = self.results
        df.to_excel(writer, sheet_name='Dashboard', index=False, startrow=1)
        ws = writer.sheets['Dashboard']
        
        # Apply headers
        for col_num, value in enumerate(df.columns.values):
            ws.write(0, col_num, value, fmt_header)
            
        # Apply column formats
        ws.set_column('B:E', 15, fmt_currency)  # $ Billions columns
        ws.set_column('F:F', 10, fmt_currency)  # EPS
        ws.set_column('G:G', 15, fmt_percent)   # EPS Accretion
        ws.set_column('H:H', 12, fmt_percent)   # Margin
        ws.set_column('J:J', 10, fmt_percent)   # ROI
        ws.set_column('A:A', 20, fmt_bold)      # Scenario names
        
        # Conditional Formatting for ROI (Green if > 0)
        ws.conditional_format('J2:J10', {'type': 'cell',
                                         'criteria': '>',
                                         'value': 0,
                                         'format': workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})})
                                         
        ws.conditional_format('J2:J10', {'type': 'cell',
                                         'criteria': '<=',
                                         'value': 0,
                                         'format': workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})})

        # Sheet 2: Assumptions
        ws_assump = workbook.add_worksheet("Assumptions")
        assumptions = [
            ["Metric", "Value"],
            ["Current Revenue", f"${self.current_revenue/1e9:.1f}B"],
            ["Current OpEx", f"${self.current_opex/1e9:.1f}B"],
            ["Tax Rate", f"{self.tax_rate*100}%"],
            ["Share Count", f"{self.shares_outstanding/1e9:.1f}B"],
            ["New Hires", self.new_headcount],
            ["Cost Per Head", f"${self.cost_per_head:,}"]
        ]
        
        for r, row in enumerate(assumptions):
            for c, val in enumerate(row):
                if r == 0:
                    ws_assump.write(r, c, val, fmt_header)
                else:
                    ws_assump.write(r, c, val)

        writer.close()
        print(f"✅ Excel Report Generated: {file_name}")

if __name__ == "__main__":
    print("--- Starting Professional Financial Model ---")
    model = NvidiaModel()
    model.run_model()
    model.generate_charts()
    model.export_excel()
    print("--- Process Complete ---")
