"""Typed Streamlit session-state helpers for OM Studio."""

from __future__ import annotations

from typing import Any

import streamlit as st

RUN_HISTORY_KEY = "run_history"
LATEST_RUN_KEY = "latest_run"


def initialize_session_state() -> None:
    st.session_state.setdefault(RUN_HISTORY_KEY, [])
    st.session_state.setdefault(LATEST_RUN_KEY, None)


def get_session_value(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)


def set_session_value(key: str, value: Any) -> None:
    st.session_state[key] = value
