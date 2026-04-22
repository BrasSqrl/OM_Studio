"""Shared-look presentation helpers for Quant Studio Monitoring."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

FINTECH_COLORWAY = [
    "#16324F",
    "#2A6F97",
    "#2C8C7B",
    "#C28A2C",
    "#A84A2A",
    "#D46A6A",
    "#607089",
    "#0E7490",
]


def apply_fintech_figure_theme(
    figure: go.Figure,
    *,
    title: str | None = None,
    height: int = 410,
) -> go.Figure:
    resolved_title = title or figure.layout.title.text or ""
    figure.update_layout(
        title=resolved_title,
        template="plotly_white",
        colorway=FINTECH_COLORWAY,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFDFC",
        margin={"l": 24, "r": 24, "t": 68, "b": 24},
        font={"family": "Aptos, Segoe UI, sans-serif", "color": "#112033"},
        title_font={"size": 20},
        legend={"orientation": "h", "y": -0.2},
    )
    figure.update_xaxes(showgrid=True, gridcolor="rgba(17, 32, 51, 0.08)")
    figure.update_yaxes(showgrid=True, gridcolor="rgba(17, 32, 51, 0.08)")
    return figure


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          .stApp {
            background:
              radial-gradient(circle at top left, rgba(42, 111, 151, 0.10), transparent 24%),
              radial-gradient(circle at top right, rgba(194, 138, 44, 0.12), transparent 22%),
              linear-gradient(180deg, #fcfaf6 0%, #f3eee5 100%);
            color: #112033;
            font-family: "Aptos", "Segoe UI", "Helvetica Neue", sans-serif;
          }
          .hero-shell { margin-bottom: 1.5rem; }
          .hero-card {
            padding: 1.9rem 2rem;
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(255, 253, 252, 0.98), rgba(246, 238, 225, 0.96));
            border: 1px solid rgba(17, 32, 51, 0.08);
            box-shadow: 0 22px 60px rgba(17, 32, 51, 0.08);
          }
          .hero-kicker {
            letter-spacing: 0.18em;
            text-transform: uppercase;
            font-size: 0.76rem;
            color: #c28a2c;
            margin-bottom: 0.45rem;
          }
          .hero-card h1 {
            margin: 0;
            color: #112033;
            font-family: "Aptos Display", "Aptos", "Segoe UI", sans-serif;
            font-size: 3rem;
            line-height: 1;
          }
          .hero-card p {
            margin-top: 0.7rem;
            margin-bottom: 0;
            color: #5f6b7a;
            font-size: 1rem;
            max-width: 58rem;
          }
          .hero-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
          }
          .hero-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.42rem 0.78rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(17, 32, 51, 0.08);
            color: #112033;
            font-size: 0.84rem;
          }
          div[data-testid="stMetric"] {
            background: rgba(255, 252, 249, 0.94);
            border: 1px solid rgba(17, 32, 51, 0.08);
            border-radius: 20px;
            box-shadow: 0 14px 32px rgba(17, 32, 51, 0.05);
            padding: 0.25rem 0.4rem;
          }
          .section-intro {
            padding: 1.25rem 1.35rem;
            border-radius: 24px;
            background: rgba(255, 252, 249, 0.92);
            border: 1px solid rgba(17, 32, 51, 0.08);
            box-shadow: 0 18px 44px rgba(17, 32, 51, 0.05);
            margin-bottom: 1rem;
          }
          .section-kicker {
            color: #c28a2c;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.76rem;
          }
          .workspace-nav-strip {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin: 1rem 0 0.75rem;
            padding: 0.1rem 0.1rem 0;
          }
          .workspace-nav-kicker {
            color: #c28a2c;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.72rem;
            margin-bottom: 0.18rem;
          }
          .workspace-nav-strip p {
            margin: 0;
            color: #5f6b7a;
            font-size: 0.92rem;
            line-height: 1.4;
          }
          .workspace-nav-tip {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.5rem 0.8rem;
            border-radius: 999px;
            background: rgba(255, 252, 249, 0.88);
            border: 1px solid rgba(17, 32, 51, 0.08);
            color: #112033;
            font-size: 0.82rem;
            white-space: nowrap;
          }
          .sidepanel-card,
          .readiness-banner,
          .outcome-card,
          .step-tracker-card,
          .empty-state-card,
          .control-deck-card,
          .decision-card,
          .tab-intro-card {
            border-radius: 24px;
            border: 1px solid rgba(17, 32, 51, 0.08);
            box-shadow: 0 18px 44px rgba(17, 32, 51, 0.05);
          }
          .sidepanel-card {
            padding: 1rem 1.05rem;
            background: rgba(255, 252, 249, 0.94);
            margin-bottom: 0.9rem;
          }
          .sidepanel-kicker,
          .readiness-kicker,
          .outcome-kicker,
          .step-tracker-kicker,
          .empty-state-kicker,
          .control-deck-kicker,
          .decision-kicker,
          .tab-intro-kicker {
            color: #c28a2c;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.72rem;
            margin-bottom: 0.35rem;
          }
          .sidepanel-card h3,
          .readiness-banner h3,
          .outcome-card h3,
          .step-tracker-card h3,
          .empty-state-card h3,
          .control-deck-card h3,
          .decision-card h3,
          .tab-intro-card h3 {
            margin: 0;
            color: #112033;
            font-size: 1.05rem;
          }
          .sidepanel-card p,
          .readiness-banner p,
          .outcome-card p,
          .step-tracker-card p,
          .empty-state-card p,
          .control-deck-card p,
          .decision-card p,
          .tab-intro-card p {
            margin: 0.5rem 0 0;
            color: #5f6b7a;
            font-size: 0.92rem;
            line-height: 1.45;
          }
          .mini-stat-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 0.5rem;
            margin-top: 0.9rem;
          }
          .mini-stat {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            align-items: center;
            gap: 0.75rem;
            padding: 0.7rem 0.8rem;
            border-radius: 18px;
            background: rgba(248, 244, 237, 0.92);
            border: 1px solid rgba(17, 32, 51, 0.06);
            box-sizing: border-box;
          }
          .mini-stat-label {
            display: block;
            min-width: 0;
            color: #5f6b7a;
            font-size: 0.78rem;
            line-height: 1.2;
            text-align: left;
            white-space: normal;
            overflow-wrap: anywhere;
          }
          .mini-stat-value {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 2rem;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            background: rgba(22, 50, 79, 0.08);
            color: #112033;
            font-size: 0.92rem;
            line-height: 1;
            white-space: nowrap;
          }
          .readiness-shell {
            position: sticky;
            top: 0.75rem;
            z-index: 5;
            margin-bottom: 1rem;
          }
          .readiness-banner {
            padding: 1rem 1.15rem;
            background: rgba(255, 252, 249, 0.96);
          }
          .readiness-banner--ready {
            background: linear-gradient(135deg, rgba(236, 247, 240, 0.98), rgba(250, 254, 251, 0.96));
            border-color: rgba(44, 140, 123, 0.28);
          }
          .readiness-banner--score_only {
            background: linear-gradient(135deg, rgba(255, 245, 225, 0.98), rgba(255, 251, 244, 0.96));
            border-color: rgba(194, 138, 44, 0.28);
          }
          .readiness-banner--blocked {
            background: linear-gradient(135deg, rgba(255, 236, 233, 0.98), rgba(255, 250, 249, 0.96));
            border-color: rgba(168, 74, 42, 0.24);
          }
          .status-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 0.85rem;
          }
          .status-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.38rem 0.72rem;
            border-radius: 999px;
            border: 1px solid transparent;
            font-size: 0.82rem;
            line-height: 1;
            white-space: nowrap;
          }
          .status-chip--pass,
          .status-chip--ready {
            background: rgba(44, 140, 123, 0.12);
            color: #1d665b;
            border-color: rgba(44, 140, 123, 0.2);
          }
          .status-chip--warning,
          .status-chip--score_only {
            background: rgba(194, 138, 44, 0.14);
            color: #8c651f;
            border-color: rgba(194, 138, 44, 0.22);
          }
          .status-chip--fail,
          .status-chip--blocked {
            background: rgba(168, 74, 42, 0.12);
            color: #8c3b24;
            border-color: rgba(168, 74, 42, 0.2);
          }
          .status-chip--na {
            background: rgba(96, 112, 137, 0.14);
            color: #465467;
            border-color: rgba(96, 112, 137, 0.2);
          }
          .outcome-card {
            padding: 1rem 1.15rem;
            background: rgba(255, 252, 249, 0.96);
            margin-bottom: 0.9rem;
          }
          .step-tracker-card,
          .empty-state-card,
          .control-deck-card,
          .decision-card,
          .tab-intro-card {
            padding: 1rem 1.15rem;
            background: rgba(255, 252, 249, 0.96);
            margin-bottom: 1rem;
          }
          .control-deck-card {
            margin-bottom: 0.75rem;
            padding: 0.8rem 1rem;
            box-shadow: 0 10px 24px rgba(17, 32, 51, 0.04);
          }
          .decision-card {
            margin-top: 0.7rem;
            box-shadow: 0 12px 28px rgba(17, 32, 51, 0.05);
          }
          .decision-card--ready,
          .decision-card--completed {
            background: linear-gradient(135deg, rgba(236, 247, 240, 0.98), rgba(250, 254, 251, 0.96));
            border-color: rgba(44, 140, 123, 0.24);
          }
          .decision-card--score_only,
          .decision-card--current {
            background: linear-gradient(135deg, rgba(255, 245, 225, 0.98), rgba(255, 251, 244, 0.96));
            border-color: rgba(194, 138, 44, 0.22);
          }
          .decision-card--blocked {
            background: linear-gradient(135deg, rgba(255, 236, 233, 0.98), rgba(255, 250, 249, 0.96));
            border-color: rgba(168, 74, 42, 0.2);
          }
          .decision-grid {
            display: grid;
            grid-template-columns: minmax(0, 1fr) 280px;
            gap: 1rem;
            align-items: start;
          }
          .decision-next {
            padding: 0.9rem 0.95rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(17, 32, 51, 0.08);
          }
          .decision-next span {
            display: block;
            color: #5f6b7a;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
          }
          .decision-next strong {
            display: block;
            margin-top: 0.4rem;
            color: #112033;
            font-size: 0.92rem;
            line-height: 1.35;
          }
          .tab-intro-card {
            padding: 0.82rem 1rem;
            background: rgba(255, 252, 249, 0.78);
            box-shadow: none;
            border-color: rgba(17, 32, 51, 0.07);
          }
          div[data-testid="stTabs"] {
            margin-top: 0.2rem;
          }
          div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.55rem;
            padding: 0.45rem;
            border-radius: 24px;
            background: rgba(255, 252, 249, 0.92);
            border: 1px solid rgba(17, 32, 51, 0.08);
            box-shadow: 0 16px 34px rgba(17, 32, 51, 0.06);
            margin-bottom: 1rem;
          }
          div[data-testid="stTabs"] [data-baseweb="tab"] {
            flex: 1 1 0;
            justify-content: center;
            min-height: 3.55rem;
            padding: 0.8rem 1rem;
            border-radius: 18px;
            background: transparent;
            transition: background 0.16s ease, box-shadow 0.16s ease, transform 0.16s ease;
          }
          div[data-testid="stTabs"] [data-baseweb="tab"]:hover {
            background: rgba(22, 50, 79, 0.06);
          }
          div[data-testid="stTabs"] [data-baseweb="tab"] p {
            margin: 0;
            color: #465467;
            font-size: 1rem;
            font-weight: 700;
            letter-spacing: 0.01em;
          }
          div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, rgba(22, 50, 79, 0.12), rgba(42, 111, 151, 0.08));
            border: 1px solid rgba(22, 50, 79, 0.12);
            box-shadow: 0 10px 22px rgba(17, 32, 51, 0.07);
            transform: translateY(-1px);
          }
          div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] p {
            color: #112033;
          }
          div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            display: none;
          }
          div[data-testid="stTabs"] [role="tabpanel"] {
            padding-top: 0.15rem;
          }
          .step-tracker-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.7rem;
            margin-top: 0.9rem;
          }
          .step {
            padding: 0.85rem 0.85rem 0.8rem;
            border-radius: 18px;
            border: 1px solid rgba(17, 32, 51, 0.08);
            background: rgba(248, 244, 237, 0.92);
            min-height: 7.25rem;
          }
          .step--complete {
            background: rgba(236, 247, 240, 0.96);
            border-color: rgba(44, 140, 123, 0.24);
          }
          .step--current {
            background: rgba(255, 245, 225, 0.96);
            border-color: rgba(194, 138, 44, 0.24);
          }
          .step--blocked {
            background: rgba(255, 236, 233, 0.96);
            border-color: rgba(168, 74, 42, 0.2);
          }
          .step-number {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.7rem;
            height: 1.7rem;
            border-radius: 999px;
            background: rgba(17, 32, 51, 0.08);
            color: #112033;
            font-size: 0.82rem;
            font-weight: 600;
          }
          .step-body {
            margin-top: 0.75rem;
          }
          .step-label-row {
            display: flex;
            justify-content: space-between;
            gap: 0.5rem;
            align-items: baseline;
          }
          .step-label-row strong {
            color: #112033;
            font-size: 0.92rem;
          }
          .step-marker {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #5f6b7a;
            white-space: nowrap;
          }
          .step-body span {
            display: block;
            margin-top: 0.4rem;
            color: #5f6b7a;
            font-size: 0.8rem;
            line-height: 1.35;
          }
          .empty-state-card ul {
            margin: 0.85rem 0 0 1.1rem;
            padding: 0;
            color: #112033;
          }
          .empty-state-card li {
            margin: 0.28rem 0;
            line-height: 1.4;
          }
          .outcome-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.7rem;
            margin-top: 0.95rem;
          }
          .outcome-metric {
            padding: 0.78rem 0.8rem;
            border-radius: 18px;
            background: rgba(248, 244, 237, 0.92);
            border: 1px solid rgba(17, 32, 51, 0.06);
          }
          .outcome-metric strong {
            display: block;
            color: #112033;
            font-size: 1rem;
          }
          .outcome-metric span {
            display: block;
            margin-top: 0.2rem;
            color: #5f6b7a;
            font-size: 0.78rem;
          }
          @media (max-width: 1180px) {
            .workspace-nav-strip {
              flex-direction: column;
              align-items: flex-start;
            }
            .decision-grid {
              grid-template-columns: 1fr;
            }
            .step-tracker-grid {
              grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            div[data-testid="stTabs"] [data-baseweb="tab-list"] {
              flex-wrap: wrap;
            }
            div[data-testid="stTabs"] [data-baseweb="tab"] {
              flex: 1 1 calc(50% - 0.55rem);
            }
          }
          @media (max-width: 720px) {
            .step-tracker-grid {
              grid-template-columns: 1fr;
            }
            div[data-testid="stTabs"] [data-baseweb="tab"] {
              flex: 1 1 100%;
              min-height: 3.2rem;
            }
            div[data-testid="stTabs"] [data-baseweb="tab"] p {
              font-size: 0.95rem;
            }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <section class="hero-shell">
          <div class="hero-card">
            <div class="hero-kicker">Ongoing Monitoring Workspace</div>
            <h1>OM Studio</h1>
            <p>
              Drop approved model bundles into the workspace, drop monitoring data
              into the watched inbox, and run test-level monitoring with exportable
              reviewer-ready artifacts.
            </p>
            <div class="hero-chip-row">
              <span class="hero-chip">Compliant bundle discovery</span>
              <span class="hero-chip">Raw-data scoring only</span>
              <span class="hero-chip">Per-test pass or fail</span>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
