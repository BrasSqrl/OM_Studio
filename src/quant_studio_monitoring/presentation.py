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
        plot_bgcolor="#FFFFFF",
        margin={"l": 24, "r": 24, "t": 68, "b": 24},
        font={"family": "Aptos, Segoe UI, sans-serif", "color": "#1F2A44"},
        title_font={"size": 18},
        legend={"orientation": "h", "y": -0.2},
    )
    figure.update_xaxes(showgrid=True, gridcolor="rgba(129, 145, 168, 0.16)")
    figure.update_yaxes(showgrid=True, gridcolor="rgba(129, 145, 168, 0.16)")
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
          :root {
            --om-bg: #f4f7fb;
            --om-surface: #ffffff;
            --om-surface-muted: #f7faff;
            --om-border: #dce5f1;
            --om-border-strong: #c8d5e8;
            --om-text: #1f2a44;
            --om-muted: #6a7891;
            --om-accent: #2e6deb;
            --om-accent-strong: #1f5cd1;
            --om-accent-soft: #edf4ff;
            --om-success: #1f9d67;
            --om-success-soft: #ecf8f2;
            --om-warning-soft: #fff7e8;
            --om-danger-soft: #fff1f1;
            --om-neutral-soft: #f1f5fb;
            --om-shadow: 0 10px 28px rgba(18, 33, 64, 0.06);
            --om-shadow-tight: 0 6px 16px rgba(18, 33, 64, 0.05);
          }
          .stApp {
            background: linear-gradient(180deg, #f8fbff 0%, var(--om-bg) 100%);
            color: var(--om-text);
          }
          [data-testid="stSidebar"] {
            background: var(--om-surface);
            border-right: 1px solid var(--om-border);
          }
          [data-testid="stSidebar"] > div:first-child {
            background: var(--om-surface);
          }
          .block-container {
            max-width: 1560px;
            padding-top: 1rem;
          }
          .hero-shell {
            margin-bottom: 1rem;
          }
          .hero-card,
          .sidepanel-card,
          .readiness-banner,
          .outcome-card,
          .step-tracker-card,
          .empty-state-card,
          .control-deck-card,
          .decision-card,
          .tab-intro-card {
            background: var(--om-surface);
            border-color: var(--om-border);
            border-radius: 18px;
            box-shadow: var(--om-shadow-tight);
          }
          .hero-card {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1.25rem;
            padding: 1rem 1.2rem;
          }
          .hero-brand {
            display: flex;
            align-items: center;
            gap: 0.95rem;
            min-width: 0;
          }
          .hero-brandmark {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 2.7rem;
            height: 2.7rem;
            border-radius: 14px;
            background: linear-gradient(135deg, var(--om-accent), #4f8cff);
            color: #ffffff;
            font-size: 0.92rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            box-shadow: 0 10px 20px rgba(46, 109, 235, 0.22);
            flex: 0 0 auto;
          }
          .hero-copy {
            min-width: 0;
          }
          .hero-kicker,
          .sidepanel-kicker,
          .readiness-kicker,
          .outcome-kicker,
          .step-tracker-kicker,
          .empty-state-kicker,
          .control-deck-kicker,
          .decision-kicker,
          .tab-intro-kicker,
          .workspace-nav-kicker,
          .section-kicker {
            color: var(--om-accent);
            font-size: 0.68rem;
            letter-spacing: 0.12em;
            font-weight: 700;
          }
          .hero-card h1 {
            font-size: 1.45rem;
            line-height: 1.1;
          }
          .hero-card p,
          .sidepanel-card p,
          .decision-card p,
          .tab-intro-card p,
          .workspace-nav-strip p {
            color: var(--om-muted);
            font-size: 0.9rem;
          }
          .hero-toolbar {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            justify-content: flex-end;
          }
          .workspace-toolbar-card {
            padding: 0.78rem 0.95rem;
            margin-bottom: 0.6rem;
            border: 1px solid var(--om-border);
            border-radius: 16px;
            background: var(--om-surface);
            box-shadow: var(--om-shadow-tight);
          }
          .workspace-toolbar-kicker {
            color: var(--om-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.67rem;
            font-weight: 700;
            margin-bottom: 0.18rem;
          }
          .workspace-toolbar-card h3 {
            margin: 0;
            color: var(--om-text);
            font-size: 1rem;
            line-height: 1.2;
          }
          .workflow-stage-card {
            display: grid;
            grid-template-columns: auto minmax(0, 1fr);
            gap: 1rem;
            align-items: start;
            padding: 1rem 1.05rem;
            margin-bottom: 0.85rem;
            border: 1px solid var(--om-border);
            border-radius: 18px;
            background: var(--om-surface);
            box-shadow: var(--om-shadow-tight);
          }
          .workflow-stage-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 2rem;
            height: 2rem;
            border-radius: 12px;
            background: var(--om-accent-soft);
            color: var(--om-accent-strong);
            font-size: 0.92rem;
            font-weight: 800;
            line-height: 1;
          }
          .workflow-stage-copy {
            min-width: 0;
          }
          .workflow-stage-kicker {
            color: var(--om-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.67rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
          }
          .workflow-stage-copy h3 {
            margin: 0;
            color: var(--om-text);
            font-size: 1.08rem;
            line-height: 1.25;
          }
          .workflow-stage-copy p {
            margin: 0.35rem 0 0;
            color: var(--om-muted);
            font-size: 0.86rem;
            line-height: 1.45;
          }
          .summary-tile-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin-bottom: 0.95rem;
          }
          .summary-tile {
            padding: 0.92rem 0.95rem;
            border-radius: 16px;
            border: 1px solid var(--om-border);
            background: var(--om-surface);
            box-shadow: var(--om-shadow-tight);
          }
          .summary-tile span {
            display: block;
            color: var(--om-muted);
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
          }
          .summary-tile strong {
            display: block;
            margin-top: 0.45rem;
            color: var(--om-text);
            font-size: 1rem;
            line-height: 1.25;
          }
          .summary-tile p {
            margin: 0.3rem 0 0;
            color: var(--om-muted);
            font-size: 0.8rem;
            line-height: 1.4;
          }
          .summary-tile--ready {
            background: linear-gradient(180deg, #ffffff 0%, var(--om-success-soft) 100%);
          }
          .summary-tile--warning,
          .summary-tile--score_only {
            background: linear-gradient(180deg, #ffffff 0%, var(--om-warning-soft) 100%);
          }
          .summary-tile--blocked,
          .summary-tile--contract_failed,
          .summary-tile--execution_failed,
          .summary-tile--fail {
            background: linear-gradient(180deg, #ffffff 0%, var(--om-danger-soft) 100%);
          }
          .summary-tile--completed,
          .summary-tile--current,
          .summary-tile--neutral {
            background: linear-gradient(180deg, #ffffff 0%, var(--om-accent-soft) 100%);
          }
          .hero-chip,
          .workspace-nav-tip {
            background: var(--om-surface-muted);
            border-color: var(--om-border);
            color: var(--om-text);
            font-size: 0.8rem;
            font-weight: 600;
            box-shadow: none;
          }
          .workspace-nav-strip {
            margin-top: 0.65rem;
          }
          .mini-stat,
          .outcome-metric,
          .decision-next,
          .step {
            background: var(--om-surface-muted);
            border-color: var(--om-border);
            border-radius: 14px;
          }
          .mini-stat-value,
          .step-number {
            background: var(--om-accent-soft);
            color: var(--om-accent-strong);
          }
          .mini-stat-label,
          .step-body span,
          .step-marker,
          .outcome-metric span,
          .decision-next span {
            color: var(--om-muted);
          }
          .status-chip {
            border-color: var(--om-border);
            font-size: 0.78rem;
            font-weight: 600;
          }
          .status-chip--pass,
          .status-chip--ready {
            background: var(--om-success-soft);
            color: #1c7c52;
            border-color: rgba(31, 157, 103, 0.18);
          }
          .status-chip--warning,
          .status-chip--score_only {
            background: var(--om-warning-soft);
            color: #8d6117;
            border-color: rgba(174, 116, 20, 0.18);
          }
          .status-chip--fail,
          .status-chip--blocked {
            background: var(--om-danger-soft);
            color: #a84848;
            border-color: rgba(195, 79, 79, 0.18);
          }
          .status-chip--na {
            background: var(--om-neutral-soft);
            color: #55657e;
            border-color: rgba(129, 145, 168, 0.18);
          }
          .decision-card--ready,
          .decision-card--completed,
          .readiness-banner--ready,
          .step--complete {
            background: linear-gradient(180deg, #ffffff 0%, var(--om-success-soft) 100%);
          }
          .decision-card--score_only,
          .decision-card--current,
          .step--current {
            background: linear-gradient(180deg, #ffffff 0%, var(--om-accent-soft) 100%);
          }
          .decision-card--blocked,
          .readiness-banner--blocked,
          .step--blocked {
            background: linear-gradient(180deg, #ffffff 0%, var(--om-danger-soft) 100%);
          }
          div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            padding: 0.15rem 0;
            border-radius: 0;
            background: transparent;
            border: none;
            box-shadow: none;
            gap: 0.5rem;
          }
          div[data-testid="stTabs"] [data-baseweb="tab"] {
            flex: 0 0 auto;
            min-height: 2.5rem;
            padding: 0.48rem 0.82rem;
            border-radius: 10px;
            background: var(--om-surface);
            border: 1px solid var(--om-border);
          }
          div[data-testid="stTabs"] [data-baseweb="tab"]:hover {
            background: #edf3fb;
          }
          div[data-testid="stTabs"] [data-baseweb="tab"] p {
            color: var(--om-muted);
            font-size: 0.82rem;
          }
          div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
            background: var(--om-accent-soft);
            border-color: rgba(46, 109, 235, 0.18);
            box-shadow: none;
          }
          div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] p {
            color: var(--om-accent-strong);
          }
          .stButton > button,
          .stDownloadButton > button {
            min-height: 2.7rem;
            border-radius: 12px;
            border: 1px solid var(--om-border);
            background: var(--om-surface);
            color: var(--om-text);
            font-weight: 700;
          }
          .stButton > button[kind="primary"],
          .stDownloadButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--om-accent), #4f8cff);
            color: #ffffff;
            border-color: rgba(46, 109, 235, 0.26);
            box-shadow: 0 10px 22px rgba(46, 109, 235, 0.18);
          }
          .stButton > button:disabled,
          .stDownloadButton > button:disabled {
            background: #f3f6fb;
            color: #8a96a8;
          }
          .stSelectbox > label,
          .stTextInput > label,
          .stNumberInput > label,
          .stDateInput > label,
          .stMultiSelect > label,
          .stToggle > label,
          .stRadio > label {
            color: var(--om-muted);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
          }
          div[data-baseweb="select"] > div,
          .stTextInput input,
          .stNumberInput input,
          .stDateInput input,
          .stMultiSelect [data-baseweb="select"] > div {
            border-radius: 14px !important;
            border-color: var(--om-border) !important;
            background: var(--om-surface) !important;
            min-height: 3.15rem;
            box-shadow: none !important;
          }
          div[data-baseweb="select"] > div:hover,
          .stTextInput input:hover,
          .stNumberInput input:hover,
          .stDateInput input:hover,
          .stMultiSelect [data-baseweb="select"] > div:hover {
            border-color: var(--om-border-strong) !important;
          }
          div[data-testid="stDataFrame"],
          div[data-testid="stTable"],
          div[data-testid="stExpander"] details,
          div[data-testid="stAlert"] {
            border: 1px solid var(--om-border);
            border-radius: 14px;
            background: var(--om-surface);
            box-shadow: var(--om-shadow-tight);
          }
          div[data-testid="stExpander"] summary {
            color: var(--om-text);
            font-weight: 700;
          }
          @media (max-width: 1180px) {
            .hero-card {
              flex-direction: column;
              align-items: flex-start;
            }
            .hero-toolbar {
              justify-content: flex-start;
            }
            .summary-tile-grid {
              grid-template-columns: 1fr;
            }
            div[data-testid="stTabs"] [data-baseweb="tab-list"] {
              flex-wrap: wrap;
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
            <div class="hero-brand">
              <div class="hero-brandmark">OM</div>
              <div class="hero-copy">
                <div class="hero-kicker">Ongoing Monitoring Workspace</div>
                <h1>OM Studio</h1>
                <p>
                  Configure, validate, and run ongoing monitoring from approved model bundles with reviewer-ready outputs.
                </p>
              </div>
            </div>
            <div class="hero-toolbar">
              <span class="hero-chip">Bundle Registry</span>
              <span class="hero-chip">Raw-Data Monitoring</span>
              <span class="hero-chip">Reviewer Outputs</span>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
