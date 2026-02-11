"""
NVIDIA Corporation — Institutional Financial Model
===================================================
Headcount Expansion ROI & Equity Valuation Analysis

All assumptions are explicitly disclosed.
All valuation outputs use a single, consistent two-stage DCF methodology.
The hiring decision's value creation is linked to its impact on equity value.

Author: Antigravity Institutional Research
Date:   Feb 11, 2026
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import xlsxwriter

# ============================================================================
# CONFIGURATION
# ============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
plt.rcParams['font.family'] = 'sans-serif'

COLORS = {
    'green':  '#27ae60',
    'red':    '#c0392b',
    'blue':   '#2980b9',
    'gray':   '#7f8c8d',
    'dark':   '#2c3e50',
    'orange': '#e67e22',
}


class NvidiaInstitutionalModel:
    """
    Institutional-grade model answering two linked questions:

      Q1 (Operating): Should NVIDIA hire 2,000 engineers?
         → Incremental ROIC with 3-year ramp, capex, working capital

      Q2 (Equity):    Is the stock worth buying at $190?
         → 10-year two-stage DCF with disclosed terminal assumptions

    The ROIC from Q1 is linked to Q2 via its enterprise value impact.
    The scenario matrix uses the same DCF function as the base case,
    ensuring internal consistency.
    """

    def __init__(self):
        # ==============================================================
        # A. MARKET DATA (as of Feb 11, 2026)
        # ==============================================================
        self.share_price = 190.00
        self.shares_out  = 24.6e9
        self.market_cap  = self.share_price * self.shares_out  # ~$4.67T

        # ==============================================================
        # B. FUNDAMENTAL BASELINE (FY26 Consensus Estimates)
        #    Source: Bloomberg / FactSet Consensus
        # ==============================================================
        self.revenue     = 213.35e9   # FY26E Revenue
        self.gross_margin = 0.745     # 74.5% GM (Data Center mix)
        self.op_margin   = 0.620      # 62.0% Operating Margin
        self.tax_rate    = 0.145      # 14.5% effective tax rate
        self.depr_pct    = 0.025      # 2.5% of rev (maintenance capex)

        # ==============================================================
        # C. WACC DERIVATION — CAPM (Fully Disclosed)
        #    Cost of Equity = Rf + β × ERP
        #    WACC = E/V × Ke + D/V × Kd × (1 − t)
        # ==============================================================
        self.risk_free       = 0.043   # 10-yr US Treasury, Feb 2026
        self.beta            = 1.50    # Bloomberg 2Y adjusted beta
        self.erp             = 0.048   # Equity Risk Premium (Damodaran)
        self.cost_of_equity  = self.risk_free + self.beta * self.erp
        self.debt_to_capital = 0.03    # NVDA is ~97% equity funded
        self.cost_of_debt    = 0.045   # Pre-tax cost of debt
        self.wacc = (
            (1 - self.debt_to_capital) * self.cost_of_equity +
            self.debt_to_capital * self.cost_of_debt * (1 - self.tax_rate)
        )
        # WACC ≈ 11.3%

        # ==============================================================
        # D. TERMINAL ASSUMPTIONS (Disclosed)
        # ==============================================================
        self.terminal_growth  = 0.030  # 3.0% perpetuity growth
        self.terminal_margin  = 0.45   # 45% steady-state op margin
        self.growth_years     = 5      # Stage 1: high-growth duration
        self.fade_years       = 5      # Stage 2: linear fade to terminal

        # ==============================================================
        # E. HEADCOUNT INVESTMENT (Detailed Mechanics)
        # ==============================================================
        self.new_hires         = 2_000
        self.cost_per_head     = 375_000        # Fully-loaded
        self.annual_opex       = self.new_hires * self.cost_per_head
        self.incr_capex        = 100e6          # Compute infra (one-time)
        self.rev_per_eng       = 750_000        # Steady-state per engineer
        self.marginal_cogs     = 0.255          # 25.5% (inverse of 74.5% GM)
        self.wc_pct            = 0.04           # Working capital % of rev
        self.ramp = {1: 0.40, 2: 0.70, 3: 1.0} # 3-year productivity ramp

        # Storage
        self.roi_detail = None
        self.base_dcf   = None

    # ==================================================================
    # WACC Summary
    # ==================================================================
    def wacc_summary(self):
        return {
            'Risk-Free Rate':       self.risk_free,
            'Beta (2Y Adjusted)':   self.beta,
            'Equity Risk Premium':  self.erp,
            'Cost of Equity':       self.cost_of_equity,
            'Debt / Capital':       self.debt_to_capital,
            'Pre-Tax Cost of Debt': self.cost_of_debt,
            'WACC':                 self.wacc,
        }

    # ==================================================================
    # Q1: HEADCOUNT ROI (Transparent Mechanics)
    # ==================================================================
    def run_headcount_roi(self):
        """
        Calculates ROIC with full disclosure:
          - Revenue per engineer with 3-year ramp schedule
          - COGS at marginal rate (25.5%)
          - One-time incremental capex ($100M)
          - Working capital at 4% of incremental revenue
          - 5-year FCF profile
          - Steady-state ROIC vs WACC
          - Enterprise value impact via perpetuity on steady NOPAT
        """
        yearly = []
        for yr in range(1, 6):
            ramp = self.ramp.get(yr, 1.0)
            rev  = self.new_hires * self.rev_per_eng * ramp
            cogs = rev * self.marginal_cogs
            gp   = rev - cogs
            opex = self.annual_opex
            oi   = gp - opex
            nopat = oi * (1 - self.tax_rate)
            wc   = rev * self.wc_pct
            capex = self.incr_capex if yr == 1 else 0
            fcf  = nopat - capex - wc

            yearly.append({
                'Year': yr, 'Ramp': f'{ramp:.0%}',
                'Revenue ($M)':      round(rev / 1e6, 1),
                'COGS ($M)':         round(cogs / 1e6, 1),
                'Gross Profit ($M)': round(gp / 1e6, 1),
                'OpEx ($M)':         round(opex / 1e6, 1),
                'NOPAT ($M)':        round(nopat / 1e6, 1),
                'CapEx ($M)':        round(capex / 1e6, 1),
                'Work Cap ($M)':     round(wc / 1e6, 1),
                'FCF ($M)':          round(fcf / 1e6, 1),
            })

        # Steady-state ROIC (Year 3+, full productivity)
        ss_nopat   = yearly[2]['NOPAT ($M)'] * 1e6
        invested   = self.annual_opex + self.incr_capex
        ss_roic    = ss_nopat / invested

        # NPV of 5-year FCF
        npv = sum(
            y['FCF ($M)'] * 1e6 / (1 + self.wacc) ** y['Year']
            for y in yearly
        )

        # Enterprise value impact (perpetuity on steady NOPAT, PV from Yr 3)
        ev_impact    = ss_nopat * (1 + self.terminal_growth) / \
                       (self.wacc - self.terminal_growth)
        ev_impact_pv = ev_impact / (1 + self.wacc) ** 3
        share_impact = ev_impact_pv / self.shares_out

        self.roi_detail = {
            'yearly':       yearly,
            'steady_roic':  ss_roic,
            'npv':          npv,
            'ev_impact_pv': ev_impact_pv,
            'share_impact': share_impact,
            'verdict':      'APPROVED' if ss_roic > self.wacc else 'REJECTED',
        }
        return self.roi_detail

    # ==================================================================
    # CORE VALUATION: Two-Stage DCF (Single Consistent Methodology)
    # ==================================================================
    def dcf_valuation(self, revenue_cagr, op_margin,
                      terminal_margin=None, terminal_growth=None):
        """
        10-year, two-stage DCF:
          Stage 1 (Yrs 1–5):  Explicit growth at revenue_cagr, stated margin
          Stage 2 (Yrs 6–10): Linear fade of growth & margin to terminal
          Terminal:            Gordon Growth on Year 10 FCF

        This function is used for:
          - Base case valuation
          - Scenario matrix
          - Reverse-engineering implied expectations
        Ensuring complete internal consistency.
        """
        tm = terminal_margin or self.terminal_margin
        tg = terminal_growth or self.terminal_growth

        cashflows = []
        prev_rev  = self.revenue

        for yr in range(1, 11):
            if yr <= self.growth_years:
                g = revenue_cagr
                m = op_margin
            else:
                fade = (yr - self.growth_years) / self.fade_years
                g = revenue_cagr * (1 - fade) + tg * fade
                m = op_margin * (1 - fade) + tm * fade

            rev   = prev_rev * (1 + g)
            nopat = rev * m * (1 - self.tax_rate)
            reinv = rev * self.depr_pct
            fcf   = nopat - reinv
            pv    = fcf / (1 + self.wacc) ** yr

            cashflows.append({
                'year': yr, 'revenue': rev, 'growth': g,
                'margin': m, 'nopat': nopat, 'fcf': fcf, 'pv_fcf': pv,
            })
            prev_rev = rev

        # Terminal value on Year 10 FCF
        yr10_fcf  = cashflows[-1]['fcf']
        tv        = yr10_fcf * (1 + tg) / (self.wacc - tg)
        pv_tv     = tv / (1 + self.wacc) ** 10

        pv_s1 = sum(c['pv_fcf'] for c in cashflows[:5])
        pv_s2 = sum(c['pv_fcf'] for c in cashflows[5:])
        ev    = pv_s1 + pv_s2 + pv_tv
        ps    = ev / self.shares_out

        return {
            'yearly':     cashflows,
            'pv_stage1':  pv_s1,
            'pv_stage2':  pv_s2,
            'tv':         tv,
            'pv_tv':      pv_tv,
            'ev':         ev,
            'per_share':  ps,
            'premium':    (self.share_price / ps - 1) if ps > 0 else None,
        }

    # ==================================================================
    # REVERSE-ENGINEER: What CAGR justifies $190?
    # ==================================================================
    def implied_cagr(self):
        lo, hi = 0.0, 0.80
        for _ in range(60):
            mid = (lo + hi) / 2
            if self.dcf_valuation(mid, self.op_margin)['per_share'] < self.share_price:
                lo = mid
            else:
                hi = mid
        rev_5y = self.revenue * (1 + mid) ** 5
        return {'cagr': mid, 'rev_5y': rev_5y}

    # ==================================================================
    # CHART 1: Investment Decision (Ramp + ROIC vs WACC)
    # ==================================================================
    def chart_investment(self):
        roi = self.roi_detail
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})

        # --- Left panel: 5-year ramp ---
        years = [y['Year'] for y in roi['yearly']]
        revs  = [y['Revenue ($M)'] for y in roi['yearly']]
        nops  = [y['NOPAT ($M)'] for y in roi['yearly']]

        x = np.arange(len(years))
        w = 0.32
        ax1.bar(x - w/2, revs, w, label='Revenue', color=COLORS['blue'],
                alpha=0.85, edgecolor='white')
        ax1.bar(x + w/2, nops, w, label='NOPAT', color=COLORS['green'],
                alpha=0.85, edgecolor='white')
        ax1.axhline(750, color=COLORS['red'], ls='--', lw=1.5,
                    label='Annual OpEx ($750M)')

        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Yr {y}' for y in years])
        ax1.set_ylabel('$ Millions')
        ax1.set_title('Headcount Investment: Revenue Ramp & Profitability',
                       fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9, loc='upper left')
        ax1.spines[['top', 'right']].set_visible(False)

        for i, (r, n) in enumerate(zip(revs, nops)):
            ax1.text(i - w/2, r + 15, f'${r:.0f}M', ha='center', fontsize=9)
            ax1.text(i + w/2, max(n, 0) + 15, f'${n:.0f}M', ha='center',
                     fontsize=9, color=COLORS['green'] if n > 0 else COLORS['red'])

        # --- Right panel: ROIC vs WACC ---
        roic = roi['steady_roic'] * 100
        wacc = self.wacc * 100
        bars = ax2.bar(
            ['WACC\n(Hurdle)', 'Steady-State\nROIC'],
            [wacc, roic],
            color=[COLORS['gray'], COLORS['green']],
            width=0.50, edgecolor='white', linewidth=1.5)

        for bar, v in zip(bars, [wacc, roic]):
            ax2.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                     f'{v:.1f}%', ha='center', fontweight='bold', fontsize=13)

        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Return vs Cost of Capital', fontsize=13, fontweight='bold')
        ax2.set_ylim(0, max(roic, wacc) * 1.35)
        ax2.spines[['top', 'right']].set_visible(False)

        fig.suptitle(
            f'Verdict: {roi["verdict"]}  •  '
            f'EV Impact: ${roi["ev_impact_pv"]/1e9:.1f}B  •  '
            f'Share Impact: +${roi["share_impact"]:.2f}',
            fontsize=11, color=COLORS['dark'], y=0.02)
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig('visual_1_investment_decision.png', dpi=300,
                    bbox_inches='tight')
        plt.close(fig)

    # ==================================================================
    # CHART 2: Valuation Bridge (DCF Build-Up vs Market)
    # ==================================================================
    def chart_valuation(self):
        b = self.base_dcf
        ps1 = b['pv_stage1'] / self.shares_out
        ps2 = b['pv_stage2'] / self.shares_out
        ptv = b['pv_tv']     / self.shares_out
        fv  = b['per_share']
        mp  = self.share_price

        fig, ax = plt.subplots(figsize=(11, 6))

        labels = [
            'Stage 1\nYrs 1–5\n(20% CAGR)',
            'Stage 2\nYrs 6–10\n(Fade)',
            'Terminal\nValue\n(3% perp.)',
            'DCF Fair\nValue',
            'Market\nPrice',
        ]
        vals = [ps1, ps2, ptv, fv, mp]
        clrs = [COLORS['blue'], COLORS['blue'], COLORS['orange'],
                COLORS['green'], COLORS['gray']]

        bars = ax.bar(labels, vals, color=clrs, width=0.52,
                      edgecolor='white', linewidth=1.5)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 2,
                    f'${v:.0f}', ha='center', fontweight='bold', fontsize=12)

        # Annotate the premium gap
        if mp > fv:
            gap = mp - fv
            mid_y = fv + gap / 2
            ax.annotate(
                f'${gap:.0f} Premium ({gap/fv:.0%})',
                xy=(4, mid_y), fontsize=12, color=COLORS['red'],
                fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', fc='#fadbd8', ec=COLORS['red']))

        ax.set_ylabel('$ Per Share', fontsize=12)
        ax.set_title(
            f'DCF Valuation Build-Up vs. Market Price\n'
            f'(WACC = {self.wacc:.1%}  |  Terminal Growth = '
            f'{self.terminal_growth:.0%}  |  Terminal Margin = '
            f'{self.terminal_margin:.0%})',
            fontsize=13, fontweight='bold', color=COLORS['dark'])
        ax.spines[['top', 'right']].set_visible(False)

        fig.tight_layout()
        fig.savefig('visual_2_valuation_bridge.png', dpi=300,
                    bbox_inches='tight')
        plt.close(fig)

    # ==================================================================
    # CHART 3: Scenario Matrix (Uses same DCF — Internally Consistent)
    # ==================================================================
    def chart_scenario_matrix(self):
        growth_rates = [0.10, 0.15, 0.20, 0.25, 0.30]
        margins      = [0.45, 0.50, 0.55, 0.60, 0.65]

        matrix = []
        for m in margins:
            row = []
            for g in growth_rates:
                ps = self.dcf_valuation(g, m)['per_share']
                row.append(round(ps, 0))
            matrix.append(row)

        matrix.reverse()
        margins_display = list(reversed(margins))

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(
            matrix, annot=True, fmt='.0f', cmap='RdYlGn',
            center=self.share_price,
            linewidths=1.5, linecolor='white',
            xticklabels=[f'{g:.0%}' for g in growth_rates],
            yticklabels=[f'{m:.0%}' for m in margins_display],
            ax=ax, cbar_kws={'label': 'Implied Share Price ($)'})

        ax.set_xlabel('Revenue CAGR (Stage 1, Years 1–5)', fontsize=12)
        ax.set_ylabel('Operating Margin', fontsize=12)
        ax.set_title(
            f'Scenario Matrix — Implied Share Price (Consistent DCF)\n'
            f'Market Price = ${self.share_price:.0f}  |  '
            f'WACC = {self.wacc:.1%}  |  '
            f'Terminal: {self.terminal_growth:.0%} growth, '
            f'{self.terminal_margin:.0%} margin',
            fontsize=12, fontweight='bold')

        fig.tight_layout()
        fig.savefig('visual_3_scenario_matrix.png', dpi=300,
                    bbox_inches='tight')
        plt.close(fig)

    # ==================================================================
    # CHART 4: Portfolio Summary (Consolidated for Website)
    # ==================================================================
    def chart_portfolio_summary(self):
        """
        Generates a single, high-resolution dashboard-style image
        combining all three key visuals for portfolio presentation.
        """
        fig = plt.figure(figsize=(16, 12), constrained_layout=True)
        gs = fig.add_gridspec(2, 2)

        # 1. ROI vs WACC (Upper Left)
        ax_roi = fig.add_subplot(gs[0, 0])
        roic = self.roi_detail['steady_roic'] * 100
        wacc = self.wacc * 100
        bars = ax_roi.bar(['WACC', 'ROIC'], [wacc, roic],
                          color=[COLORS['gray'], COLORS['green']],
                          width=0.6, alpha=0.9, edgecolor='white')
        ax_roi.set_title('Strategic ROI vs. Hurdle Rate', fontweight='bold', fontsize=16)
        ax_roi.set_ylabel('Percentage (%)')
        for bar in bars:
            ax_roi.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{bar.get_height():.1f}%', ha='center', fontweight='bold')
        ax_roi.spines[['top', 'right']].set_visible(False)

        # 2. Valuation Bridge (Upper Right)
        ax_val = fig.add_subplot(gs[0, 1])
        b = self.base_dcf
        labels = ['S1', 'S2', 'TV', 'Fair', 'Market']
        vals = [b['pv_stage1']/self.shares_out, b['pv_stage2']/self.shares_out,
                b['pv_tv']/self.shares_out, b['per_share'], self.share_price]
        clrs = [COLORS['blue'], COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['gray']]
        bars = ax_val.bar(labels, vals, color=clrs, alpha=0.9, edgecolor='white')
        ax_val.set_title('DCF Build-Up vs. Market Price', fontweight='bold', fontsize=16)
        ax_val.set_ylabel('Price per Share ($)')
        for bar, v in zip(bars, vals):
            ax_val.text(bar.get_x() + bar.get_width()/2, v + 2, f'${v:.0f}', ha='center', fontsize=10)
        ax_val.spines[['top', 'right']].set_visible(False)

        # 3. Scenario Matrix (Bottom Span)
        ax_mat = fig.add_subplot(gs[1, :])
        growth_rates = [0.10, 0.15, 0.20, 0.25, 0.30]
        margins = [0.45, 0.50, 0.55, 0.60, 0.65]
        matrix = []
        for m in margins:
            row = [self.dcf_valuation(g, m)['per_share'] for g in growth_rates]
            matrix.append(row)
        matrix.reverse()
        sns.heatmap(matrix, annot=True, fmt='.0f', cmap='RdYlGn', center=self.share_price,
                    xticklabels=[f'{g:.0%}' for g in growth_rates],
                    yticklabels=[f'{m:.0%}' for m in reversed(margins)],
                    ax=ax_mat, cbar=False, linewidths=1)
        ax_mat.set_title('Valuation Sensitivity Matrix (Growth vs. Margin)', fontweight='bold', fontsize=18)
        ax_mat.set_xlabel('Revenue CAGR (Years 1-5)')
        ax_mat.set_ylabel('Operating Margin')

        plt.suptitle('NVIDIA Strategic Financial Analysis Dashboard', fontsize=24, fontweight='bold', y=1.05)
        fig.savefig('nvidia_portfolio_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    # ==================================================================
    # EXCEL EXPORT (Full Disclosure — 4 Sheets)
    # ==================================================================
    def export_excel(self):
        writer = pd.ExcelWriter('Nvidia_Final_Model.xlsx', engine='xlsxwriter')
        wb = writer.book

        hdr = wb.add_format({
            'bold': True, 'bg_color': '#2c3e50', 'font_color': 'white',
            'border': 1, 'font_size': 11})
        pct = wb.add_format({'num_format': '0.0%', 'border': 1})
        txt = wb.add_format({'border': 1, 'font_size': 11})

        # ---- Sheet 1: Assumptions ----
        ws = wb.add_worksheet('Assumptions')
        writer.sheets['Assumptions'] = ws
        ws.set_column('A:A', 35); ws.set_column('B:B', 18)
        ws.set_column('C:C', 42)

        for c, h in enumerate(['Parameter', 'Value', 'Source / Rationale']):
            ws.write(0, c, h, hdr)

        assumptions = [
            ('Share Price',                self.share_price,        'Market data, Feb 11 2026'),
            ('Shares Outstanding (B)',     f'{self.shares_out/1e9:.1f}', 'Latest 10-Q'),
            ('Revenue FY26E ($B)',         f'{self.revenue/1e9:.1f}',    'Bloomberg Consensus'),
            ('Gross Margin',               self.gross_margin,       'FY25 10-K, DC mix'),
            ('Operating Margin',           self.op_margin,          'FY25 10-K'),
            ('Effective Tax Rate',         self.tax_rate,           'FY25 10-K'),
            ('Maint. CapEx (% Rev)',       self.depr_pct,           'Depreciation proxy'),
            ('', '', ''),
            ('— WACC (CAPM) —', '', ''),
            ('Risk-Free Rate',             self.risk_free,          '10-Year US Treasury'),
            ('Beta (2Y Adjusted)',         self.beta,               'Bloomberg Terminal'),
            ('Equity Risk Premium',        self.erp,                'Damodaran (Jan 2026)'),
            ('Cost of Equity',             self.cost_of_equity,     'Rf + β × ERP'),
            ('Debt / Capital',             self.debt_to_capital,    'Balance sheet'),
            ('WACC',                       self.wacc,               'Blended cost of capital'),
            ('', '', ''),
            ('— Terminal —', '', ''),
            ('Terminal Growth',            self.terminal_growth,    'GDP + inflation proxy'),
            ('Terminal Op Margin',         self.terminal_margin,    'Sector mean reversion'),
            ('Growth Phase (Yrs)',         self.growth_years,       'Consensus duration'),
            ('Fade Phase (Yrs)',           self.fade_years,         'Linear fade'),
            ('', '', ''),
            ('— Headcount —', '', ''),
            ('New Hires',                  self.new_hires,          'Management guidance'),
            ('Cost / Head (Loaded)',       f'${self.cost_per_head:,.0f}', 'Salary+SBC+overhead'),
            ('Rev / Engineer (Steady)',    f'${self.rev_per_eng:,.0f}',   'Historical rev/employee'),
            ('Ramp: Year 1',              '40%',                   'Onboarding lag'),
            ('Ramp: Year 2',              '70%',                   'Partial productivity'),
            ('Ramp: Year 3+',             '100%',                  'Full contribution'),
            ('Marginal COGS',             self.marginal_cogs,      'Inverse of 74.5% GM'),
            ('Incr. CapEx',               f'${self.incr_capex/1e6:.0f}M', 'Compute infra'),
            ('Working Capital (% Rev)',    self.wc_pct,             'DSO−DPO proxy'),
        ]
        for r, (lbl, val, src) in enumerate(assumptions, 1):
            ws.write(r, 0, lbl, txt)
            if isinstance(val, float) and val < 1:
                ws.write(r, 1, val, pct)
            else:
                ws.write(r, 1, str(val) if not isinstance(val, (int, float)) else val, txt)
            ws.write(r, 2, src, txt)

        # ---- Sheet 2: Headcount ROI ----
        roi_df = pd.DataFrame(self.roi_detail['yearly'])
        roi_df.to_excel(writer, sheet_name='Headcount ROI', index=False)
        ws2 = writer.sheets['Headcount ROI']
        n = len(roi_df)
        for r, (lbl, val) in enumerate([
            ('Steady-State ROIC',        f"{self.roi_detail['steady_roic']:.1%}"),
            ('WACC (Hurdle)',            f'{self.wacc:.1%}'),
            ('Verdict',                  self.roi_detail['verdict']),
            ('EV Impact (PV)',           f"${self.roi_detail['ev_impact_pv']/1e9:.2f}B"),
            ('Share Price Impact',       f"+${self.roi_detail['share_impact']:.2f}"),
        ], start=n + 2):
            ws2.write(r, 0, lbl, txt)
            ws2.write(r, 1, val, txt)

        # ---- Sheet 3: DCF Valuation ----
        b = self.base_dcf
        dcf_rows = [{
            'Year':            y['year'],
            'Revenue ($B)':    round(y['revenue'] / 1e9, 1),
            'Growth':          f"{y['growth']:.1%}",
            'Op Margin':       f"{y['margin']:.1%}",
            'NOPAT ($B)':      round(y['nopat'] / 1e9, 1),
            'FCF ($B)':        round(y['fcf'] / 1e9, 1),
            'PV of FCF ($B)':  round(y['pv_fcf'] / 1e9, 1),
        } for y in b['yearly']]
        dcf_df = pd.DataFrame(dcf_rows)
        dcf_df.to_excel(writer, sheet_name='DCF Valuation', index=False)
        ws3 = writer.sheets['DCF Valuation']
        r = len(dcf_df) + 2
        for i, (lbl, val) in enumerate([
            ('PV Stage 1 (Yrs 1–5)',  f"${b['pv_stage1']/1e9:.1f}B"),
            ('PV Stage 2 (Yrs 6–10)', f"${b['pv_stage2']/1e9:.1f}B"),
            ('PV Terminal Value',     f"${b['pv_tv']/1e9:.1f}B"),
            ('Enterprise Value',      f"${b['ev']/1e9:.1f}B"),
            ('Implied Share Price',   f"${b['per_share']:.2f}"),
            ('Market Price',          f"${self.share_price:.2f}"),
            ('Premium / (Discount)',  f"{b['premium']:.1%}" if b['premium'] else 'N/A'),
        ]):
            ws3.write(r + i, 0, lbl, txt)
            ws3.write(r + i, 1, val, txt)

        # ---- Sheet 4: Scenario Matrix ----
        growth_rates = [0.10, 0.15, 0.20, 0.25, 0.30]
        margins_list = [0.45, 0.50, 0.55, 0.60, 0.65]
        rows = []
        for m in margins_list:
            row = {'Op Margin': f'{m:.0%}'}
            for g in growth_rates:
                row[f'{g:.0%} CAGR'] = round(
                    self.dcf_valuation(g, m)['per_share'], 0)
            rows.append(row)
        pd.DataFrame(rows).to_excel(
            writer, sheet_name='Scenario Matrix', index=False)

        writer.close()
        print('  → Nvidia_Final_Model.xlsx (4 sheets)')

    # ==================================================================
    # RUN ALL
    # ==================================================================
    def run(self):
        sep = '=' * 58
        print(sep)
        print('  NVIDIA INSTITUTIONAL FINANCIAL MODEL')
        print(sep)

        print('\n[1/7] WACC Derivation …')
        w = self.wacc_summary()
        print(f'      Rf = {w["Risk-Free Rate"]:.1%}  '
              f'β = {w["Beta (2Y Adjusted)"]:.2f}  '
              f'ERP = {w["Equity Risk Premium"]:.1%}')
        print(f'      Cost of Equity = {w["Cost of Equity"]:.2%}')
        print(f'      WACC = {w["WACC"]:.2%}')

        print('\n[2/7] Headcount ROI …')
        roi = self.run_headcount_roi()
        print(f'      Steady-State ROIC = {roi["steady_roic"]:.1%}')
        print(f'      Verdict: {roi["verdict"]}')
        print(f'      EV Impact: ${roi["ev_impact_pv"]/1e9:.1f}B')
        print(f'      Share Impact: +${roi["share_impact"]:.2f}')

        print('\n[3/7] Base-Case DCF (20% CAGR, 62% margin) …')
        self.base_dcf = self.dcf_valuation(0.20, self.op_margin)
        b = self.base_dcf
        print(f'      DCF Fair Value = ${b["per_share"]:.2f}')
        print(f'      Market Price   = ${self.share_price:.2f}')
        print(f'      Premium        = {b["premium"]:.1%}')

        print('\n[4/7] Implied Expectations …')
        imp = self.implied_cagr()
        print(f'      CAGR to justify $190: {imp["cagr"]:.1%}')
        print(f'      Implied 5Y Revenue:   ${imp["rev_5y"]/1e9:.0f}B')

        print('\n[5/7] Generating visuals …')
        self.chart_investment()
        print('      ✓ visual_1_investment_decision.png')
        self.chart_valuation()
        print('      ✓ visual_2_valuation_bridge.png')
        self.chart_scenario_matrix()
        print('      ✓ visual_3_scenario_matrix.png')

        print('\n[6/7] Generating Website Portfolio Dashboard …')
        self.chart_portfolio_summary()
        print('      ✓ nvidia_portfolio_dashboard.png')

        print('\n[7/7] Exporting Excel …')
        self.export_excel()

        print(f'\n{sep}')
        print('  ✅  Complete.')
        print(sep)


if __name__ == '__main__':
    NvidiaInstitutionalModel().run()
