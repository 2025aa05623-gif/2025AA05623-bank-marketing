"""Compatibility layer used by the Streamlit app.

This module forwards data loading and preprocessing calls to the
implementation in `src.data` so `app.py` can import `load_data`,
`show_sample`, and `preprocess_data` from `data` as expected.
"""

import streamlit as st
from src.data import load_data as src_load_data, show_sample as src_show_sample, preprocess_data as src_preprocess


def load_data(source="url", uploaded_file=None):
    return src_load_data(source=source, uploaded_file=uploaded_file)


def show_sample(df, n=10):
    return src_show_sample(df, rows=n)


def preprocess_data(df, target_column='y'):
    return src_preprocess(df, target_column=target_column)
