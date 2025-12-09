"""
Streamlit dashboard: Myanmar Food Price Intelligence Dashboard (2008–2025)

The app tells a presentation-style story with interactive visuals covering:
- National trends, category insights, regional inequality, shock detection.
- Rice market deep-dive with affordability and anomaly checks.
- Forward-looking forecast and risk scoring.

Run with:
  pip install -r requirements.txt  # see notes at bottom of this file
  streamlit run dashboard_app.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

DATA_PATH = Path("clean_wfp_food_prices.csv")
EVENTS = [
    ("2015 Flood", pd.Timestamp("2015-08-01")),
    ("2020 COVID", pd.Timestamp("2020-04-01")),
    ("2021 Crisis", pd.Timestamp("2021-02-01")),
    ("2025 EQ", pd.Timestamp("2025-03-01")),
]
STAPLES = [
    "Rice (low quality)",
    "Rice (high quality)",
    "Oil (palm)",
    "Pulses",
    "Eggs (local)",
]
RICE_VARIANTS = [
    "Rice (low quality)",
    "Rice (high quality)",
    "Rice (Emata)",
]
PALETTE = [
    "#2e7d32",  # dark green
    "#66bb6a",  # bright green
    "#a5d6a7",  # mint
    "#558b2f",  # olive
    "#8bc34a",  # lime
    "#c5e1a5",  # pale lime
    "#33691e",  # forest green
]
GREEN_SCALE = ["#e8f5e9", "#c5e1a5", "#9ccc65", "#66bb6a", "#388e3c", "#2e7d32"]


def shorten_label(value: str, max_len: int = 18) -> str:
    if pd.isna(value):
        return value
    text = str(value)
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def style_fig(fig: go.Figure, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        legend=dict(x=1.08),
    )
    if title:
        fig.update_layout(title=title)
    fig.update_xaxes(tickangle=30)
    return fig


@st.cache_data(show_spinner=False)
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["year"] = df["date"].dt.year
    df["category_clean"] = df["category"].str.title()
    df["admin1_clean"] = df["admin1"].str.title()
    df["commodity_clean"] = df["commodity"].str.title()
    df["admin1_label"] = df["admin1_clean"].apply(lambda x: shorten_label(x, 18))
    df["commodity_label"] = df["commodity_clean"].apply(lambda x: shorten_label(x, 18))
    df["category_label"] = df["category_clean"].apply(lambda x: shorten_label(x, 18))
    df["rice_group"] = df["commodity"].apply(lambda x: "Rice (combined)" if "Rice" in x else None)
    return df


@dataclass
class HeroMetrics:
    inflation_pct: float
    shock_events: int
    most_volatile: str
    most_volatile_cv: float


def compute_hero_metrics(df: pd.DataFrame) -> HeroMetrics:
    monthly = df.groupby("month")["price"].mean().sort_index()
    inflation = (monthly.iloc[-1] - monthly.iloc[0]) / monthly.iloc[0] * 100

    monthly_state = (
        df.groupby(["admin1_clean", "commodity", "month"])["price"]
        .mean()
        .reset_index()
        .sort_values("month")
    )
    monthly_state["pct_change"] = monthly_state.groupby(["admin1_clean", "commodity"])[
        "price"
    ].pct_change()
    shocks = monthly_state[monthly_state["pct_change"].abs() >= 0.2]

    cv = (
        df.groupby("commodity")["price"]
        .agg(["mean", "std"])
        .assign(cv=lambda x: x["std"] / x["mean"])
        .sort_values("cv", ascending=False)
    )
    top = cv.iloc[0]
    return HeroMetrics(
        inflation_pct=float(inflation),
        shock_events=len(shocks),
        most_volatile=str(cv.index[0]),
        most_volatile_cv=float(top["cv"]),
    )


def add_event_markers(fig: go.Figure, active: bool = True) -> go.Figure:
    if not active:
        return fig
    for name, ts in EVENTS:
        # plotly add_vline computes means via Python sum(); pandas Timestamps no
        # longer add to ints, so send milliseconds epoch to avoid TypeErrors.
        x_value = pd.Timestamp(ts).value / 1_000_000
        fig.add_vline(
            x=x_value,
            line_dash="dot",
            line_color=PALETTE[3],
            opacity=0.55,
            annotation_text=name,
            annotation_position="top left",
            annotation_textangle=45,
        )
    return fig


def national_trend_plot(df: pd.DataFrame, currency: str) -> go.Figure:
    field = "price" if currency == "MMK" else "usdprice"
    monthly = df.groupby("month")[field].mean().reset_index()
    fig = px.line(
        monthly,
        x="month",
        y=field,
        template="plotly_white",
        color_discrete_sequence=[PALETTE[0]],
        labels={"month": "Month", field: f"Avg price ({currency})"},
    )
    fig.update_traces(mode="lines+markers")
    fig.update_layout(
        title=f"National average food price ({currency})",
        hovermode="x unified",
        legend=dict(x=1.08),
    )
    fig.update_xaxes(tickangle=30)
    return add_event_markers(fig, active=True)


def commodity_index_plot(df: pd.DataFrame, base_year: int) -> go.Figure:
    subset = df[df["commodity"].isin(STAPLES)].copy()
    subset["year"] = subset["date"].dt.year
    subset["month"] = subset["date"].dt.to_period("M").dt.to_timestamp()
    monthly = subset.groupby(["commodity", "month"])["price"].mean().reset_index()
    base_lookup = (
        monthly[monthly["month"].dt.year == base_year]
        .groupby("commodity")["price"]
        .mean()
    )
    monthly["base"] = monthly["commodity"].map(base_lookup)
    # Fallback to earliest observed price if base year data is missing for a commodity.
    fallback = monthly.groupby("commodity")["price"].transform("first")
    monthly["base"] = monthly["base"].fillna(fallback)
    monthly["indexed"] = monthly["price"] / monthly["base"] * 100

    fig = px.line(
        monthly,
        x="month",
        y="indexed",
        color="commodity",
        color_discrete_sequence=[
            "#2e7d32",
            "#1b9e77",
            "#d95f02",
            "#7570b3",
            "#66a61e",
            "#e6ab02",
        ],
        template="plotly_white",
        labels={"indexed": f"Index (base={base_year}=100)", "commodity": "Commodity"},
    )
    fig.update_layout(
        title=f"Staple price comparison (indexed to {base_year})",
        hovermode="x unified",
        legend_title="Staples",
        legend=dict(x=1.08),
    )
    fig.update_xaxes(tickangle=30)
    return add_event_markers(fig, active=False)


def category_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = (
        df.groupby(["category_clean", "year"])["price"]
        .mean()
        .reset_index()
        .pivot(index="category_clean", columns="year", values="price")
    )
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="YlGn",
        labels={"x": "Year", "y": "Category", "color": "Avg price (MMK)"},
    )
    fig.update_layout(title="Category-level price heatmap", template="plotly_white", legend=dict(x=1.08))
    fig.update_xaxes(tickangle=30)
    return fig


def category_change(df: pd.DataFrame) -> Tuple[go.Figure, Dict[str, float]]:
    cat_year = (
        df.groupby(["category_clean", "year"])["price"]
        .mean()
        .reset_index()
        .pivot(index="category_clean", columns="year", values="price")
    )
    first_year = cat_year.columns.min()
    last_year = cat_year.columns.max()
    change = (
        (cat_year[last_year] - cat_year[first_year]) / cat_year[first_year] * 100
    ).dropna().sort_values(ascending=False)
    fig = px.bar(
        change.reset_index(),
        x="category_clean",
        y=0,
        template="plotly_white",
        color=0,
        color_continuous_scale="YlGn",
        labels={"category_clean": "Category", "0": f"% change {first_year}–{last_year}"},
    )
    fig.update_layout(
        title="Category price change (%)",
        xaxis_tickangle=30,
        showlegend=False,
        legend=dict(x=1.08),
    )
    info = {"most_inflationary": change.index[0], "most_stable": change.index[-1]}
    return fig, info


def state_summary(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    state_avg = (
        df.groupby("admin1_clean")["price"]
        .mean()
        .sort_values(ascending=False)
        .reset_index(name="avg_price")
    )
    state_vol = (
        df.groupby("admin1_clean")["price"]
        .agg(["mean", "std"])
        .assign(cv=lambda x: x["std"] / x["mean"])
        .sort_values("cv", ascending=False)
        .reset_index()
    )
    return state_avg, state_vol


def state_map(df: pd.DataFrame) -> go.Figure:
    geo = (
        df.groupby("admin1_clean")[["latitude", "longitude", "price"]]
        .mean()
        .dropna()
        .reset_index()
    )
    fig = px.scatter_mapbox(
        geo,
        lat="latitude",
        lon="longitude",
        size="price",
        color="price",
        color_continuous_scale=GREEN_SCALE,
        hover_name="admin1_clean",
        zoom=4.2,
        mapbox_style="carto-positron",
    )
    fig.update_layout(title="Regional price map (average, bubble size=price)")
    return fig


def shocks_by_state(df: pd.DataFrame, threshold: float = 0.2) -> Tuple[pd.DataFrame, go.Figure]:
    monthly = (
        df.groupby(["admin1_clean", "commodity", "month"])["price"]
        .mean()
        .reset_index()
        .sort_values("month")
    )
    monthly["pct_change"] = monthly.groupby(["admin1_clean", "commodity"])["price"].pct_change()
    shocks = monthly[np.abs(monthly["pct_change"]) >= threshold].copy()
    shocks["year"] = shocks["month"].dt.year
    heat = (
        shocks.groupby(["admin1_clean", "year"])
        .size()
        .reset_index(name="shock_count")
        .pivot(index="admin1_clean", columns="year", values="shock_count")
        .fillna(0)
    )
    fig = px.imshow(
        heat,
        aspect="auto",
        color_continuous_scale="YlGn",
        labels={"x": "Year", "y": "State", "color": "Shock count"},
        title="Shock heatmap (|Δmonth| ≥ 20%)",
    )
    fig.update_layout(legend=dict(x=1.08))
    return shocks, fig


def commodity_volatility(df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    vol = (
        df.groupby("commodity")["price"]
        .agg(["mean", "std"])
        .assign(cv=lambda x: x["std"] / x["mean"])
        .sort_values("cv", ascending=False)
        .reset_index()
    )
    most = px.bar(
        vol.head(8),
        x="commodity",
        y="cv",
        color="cv",
        color_continuous_scale="YlGn",
        template="plotly_white",
        title="Most volatile foods (CV)",
    )
    stable = px.bar(
        vol.tail(8),
        x="commodity",
        y="cv",
        color="cv",
        color_continuous_scale="Greens",
        template="plotly_white",
        title="Most stable foods (CV)",
    )
    for fig in (most, stable):
        fig.update_layout(showlegend=False, xaxis_tickangle=30, legend=dict(x=1.08))
    return most, stable


def avg_price_by_category_fig(df: pd.DataFrame) -> go.Figure:
    cat = (
        df.groupby("category_label")["price"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig = px.bar(
        cat,
        x="category_label",
        y="price",
        color="price",
        color_continuous_scale="YlGn",
        template="plotly_white",
        title="Average price by category",
        labels={"category_label": "Category", "price": "Avg price (MMK)"},
    )
    fig.update_layout(showlegend=False, legend=dict(x=1.08))
    fig.update_xaxes(tickangle=30)
    fig.update_yaxes(tickangle=90)
    return fig


def top_commodities_recorded_fig(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    counts = (
        df.groupby("commodity_label")
        .size()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index(name="records")
    )
    fig = px.bar(
        counts,
        x="commodity_label",
        y="records",
        color="records",
        color_continuous_scale="YlGn",
        template="plotly_white",
        title=f"Top {top_n} most recorded commodities",
        labels={"commodity_label": "Commodity", "records": "Observations"},
    )
    fig.update_layout(showlegend=False, legend=dict(x=1.08))
    fig.update_xaxes(tickangle=30)
    fig.update_yaxes(tickangle=90)
    return fig


def avg_price_by_region_fig(df: pd.DataFrame) -> go.Figure:
    reg = (
        df.groupby("admin1_label")["price"]
        .mean()
        .sort_values(ascending=False)
        .reset_index(name="avg_price")
    )
    fig = px.bar(
        reg,
        x="admin1_label",
        y="avg_price",
        color="avg_price",
        color_continuous_scale="YlGn",
        template="plotly_white",
        title="Average price by region",
        labels={"admin1_label": "Region", "avg_price": "Avg price (MMK)"},
    )
    fig.update_layout(showlegend=False, legend=dict(x=1.08))
    fig.update_xaxes(tickangle=30)
    fig.update_yaxes(tickangle=90)
    return fig


def price_distribution_by_region_fig(df: pd.DataFrame) -> go.Figure:
    order = (
        df.groupby("admin1_label")["price"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    fig = px.box(
        df,
        x="admin1_label",
        y="price",
        category_orders={"admin1_label": order},
        color_discrete_sequence=[PALETTE[2]],
        template="plotly_white",
        title="Price distribution by region",
        labels={"admin1_label": "Region", "price": "Price (MMK)"},
    )
    fig.update_layout(showlegend=False, legend=dict(x=1.08))
    fig.update_xaxes(tickangle=30)
    fig.update_yaxes(tickangle=90)
    return fig


def average_rice_by_state_year_fig(df: pd.DataFrame) -> go.Figure:
    rice = df[df["rice_group"].notna()].copy()
    pivot = (
        rice.groupby(["admin1_label", "year"])["price"]
        .mean()
        .reset_index()
        .pivot(index="admin1_label", columns="year", values="price")
    )
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="YlGn",
        labels={"x": "Year", "y": "State", "color": "Avg rice price"},
        title="Average rice price by state by year (combined rice)",
    )
    fig.update_layout(legend=dict(x=1.08))
    fig.update_xaxes(tickangle=30)
    return fig


def yoy_change_plot(df: pd.DataFrame) -> go.Figure:
    monthly = df.groupby("month")["price"].mean().sort_index()
    yoy = monthly.pct_change(12) * 100
    fig = px.line(
        yoy.reset_index(),
        x="month",
        y="price",
        color_discrete_sequence=[PALETTE[1]],
        template="plotly_white",
        labels={"month": "Month", "price": "YoY % change"},
        title="Year-on-year % change (monthly)",
    )
    fig.update_layout(legend=dict(x=1.08), hovermode="x unified")
    fig.update_xaxes(tickangle=30)
    return add_event_markers(fig, active=True)


def seasonal_heatmap(df: pd.DataFrame) -> go.Figure:
    monthly = df.groupby("month")["price"].mean().reset_index()
    monthly["month_num"] = monthly["month"].dt.month
    monthly["year"] = monthly["month"].dt.year
    pivot = monthly.pivot(index="month_num", columns="year", values="price")
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="YlGn",
        labels={"x": "Year", "y": "Month", "color": "Avg price"},
        title="Monthly seasonal heatmap",
    )
    fig.update_layout(legend=dict(x=1.08))
    fig.update_xaxes(tickangle=30)
    return fig


def pre_post_region_price_fig(df: pd.DataFrame) -> go.Figure:
    pre = df[df["year"] < 2021].groupby("admin1_label")["price"].mean()
    post = df[df["year"] >= 2021].groupby("admin1_label")["price"].mean()
    stacked = (
        pd.concat([pre.rename("Pre-2021"), post.rename("Post-2021")], axis=1)
        .dropna()
        .reset_index()
    )
    melted = stacked.melt(id_vars="admin1_label", value_name="avg_price", var_name="period")
    fig = px.bar(
        melted,
        x="admin1_label",
        y="avg_price",
        color="period",
        barmode="group",
        template="plotly_white",
        color_discrete_sequence=[PALETTE[2], PALETTE[0]],
        title="Average price by region (pre vs post 2021)",
        labels={"admin1_label": "Region", "avg_price": "Avg price (MMK)"},
    )
    fig.update_layout(legend=dict(x=1.08))
    fig.update_xaxes(tickangle=90)
    return fig


def top_post_increase_regions_fig(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    pre = df[df["year"] < 2021].groupby("admin1_label")["price"].mean()
    post = df[df["year"] >= 2021].groupby("admin1_label")["price"].mean()
    diff = (post - pre).dropna().sort_values(ascending=False).head(top_n).reset_index(name="diff")
    fig = px.bar(
        diff,
        x="admin1_label",
        y="diff",
        color="diff",
        color_continuous_scale="YlGn",
        template="plotly_white",
        title=f"Top {top_n} regions with biggest post-2021 price increase",
        labels={"admin1_label": "Region", "diff": "Post-2021 minus pre-2021 price"},
    )
    fig.update_layout(showlegend=False, legend=dict(x=1.08))
    fig.update_xaxes(tickangle=90)
    return fig


def expensive_regions_fig(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    reg = (
        df.groupby("admin1_label")["price"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index(name="avg_price")
    )
    fig = px.bar(
        reg,
        x="admin1_label",
        y="avg_price",
        color="avg_price",
        color_continuous_scale="YlGn",
        template="plotly_white",
        title=f"Top {top_n} most expensive regions (2008–2025)",
        labels={"admin1_label": "Region", "avg_price": "Avg price (MMK)"},
    )
    fig.update_layout(showlegend=False, legend=dict(x=1.08))
    fig.update_xaxes(tickangle=90)
    return fig


def regional_volatility_fig(df: pd.DataFrame) -> go.Figure:
    vol = (
        df.groupby("admin1_label")["price"]
        .agg(["mean", "std"])
        .assign(cv=lambda x: x["std"] / x["mean"])
        .sort_values("cv", ascending=False)
        .reset_index()
    )
    fig = px.bar(
        vol,
        x="admin1_label",
        y="cv",
        color="cv",
        color_continuous_scale="YlGn",
        template="plotly_white",
        title="Regional volatility (CV)",
        labels={"admin1_label": "Region", "cv": "Coefficient of variation"},
    )
    fig.update_layout(showlegend=False, legend=dict(x=1.08))
    fig.update_xaxes(tickangle=90)
    return fig


def inequality_box_fig(df: pd.DataFrame) -> go.Figure:
    highlight = {"Sagaing": "Highlight", "Mandalay": "Highlight", "Kachin": "Highlight"}
    df_box = df.copy()
    df_box["highlight"] = df_box["admin1_clean"].map(highlight).fillna("Other")
    fig = px.box(
        df_box,
        x="admin1_label",
        y="price",
        color="highlight",
        color_discrete_map={"Highlight": PALETTE[0], "Other": PALETTE[2]},
        template="plotly_white",
        labels={"admin1_label": "Region", "price": "Price (MMK)"},
        title="Price distribution inequality (Sagaing, Mandalay, Kachin highlighted)",
    )
    fig.update_layout(legend=dict(x=1.08))
    fig.update_xaxes(tickangle=90)
    return fig


def top_yearly_change_commodities_fig(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    yearly = (
        df.groupby(["commodity_label", "year"])["price"]
        .mean()
        .reset_index()
        .sort_values(["commodity_label", "year"])
    )
    records = []
    for commodity, group in yearly.groupby("commodity_label"):
        first_year = group["year"].min()
        last_year = group["year"].max()
        if first_year == last_year:
            continue
        first_price = group[group["year"] == first_year]["price"].iloc[0]
        last_price = group[group["year"] == last_year]["price"].iloc[0]
        years = last_year - first_year
        avg_change = ((last_price / first_price) ** (1 / years) - 1) * 100
        records.append((commodity, avg_change))
    top = pd.DataFrame(records, columns=["commodity_label", "avg_annual_change"]).sort_values(
        "avg_annual_change", ascending=False
    ).head(top_n)
    fig = px.bar(
        top,
        x="commodity_label",
        y="avg_annual_change",
        color="avg_annual_change",
        color_continuous_scale="YlGn",
        template="plotly_white",
        title=f"Top {top_n} commodities by average yearly price change",
        labels={"commodity_label": "Commodity", "avg_annual_change": "Avg annual change (%)"},
    )
    fig.update_layout(showlegend=False, legend=dict(x=1.08))
    fig.update_xaxes(tickangle=30)
    return fig


def inflationary_since_2021_fig(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    recent = df[df["year"] >= 2021]
    by_year = (
        recent.groupby(["commodity_label", "year"])["price"]
        .mean()
        .reset_index()
        .sort_values(["commodity_label", "year"])
    )
    records = []
    for commodity, group in by_year.groupby("commodity_label"):
        first_year = group["year"].min()
        last_year = group["year"].max()
        if first_year == last_year:
            continue
        first_price = group[group["year"] == first_year]["price"].iloc[0]
        last_price = group[group["year"] == last_year]["price"].iloc[0]
        change = (last_price - first_price) / first_price * 100
        records.append((commodity, change))
    top = pd.DataFrame(records, columns=["commodity_label", "pct_change"]).sort_values(
        "pct_change", ascending=False
    ).head(top_n)
    fig = px.bar(
        top,
        x="commodity_label",
        y="pct_change",
        color="pct_change",
        color_continuous_scale="YlGn",
        template="plotly_white",
        title=f"Top {top_n} most inflationary commodities since 2021",
        labels={"commodity_label": "Commodity", "pct_change": "% change since 2021"},
    )
    fig.update_layout(showlegend=False, legend=dict(x=1.08))
    fig.update_xaxes(tickangle=30)
    return fig


def rice_section(df: pd.DataFrame) -> Dict[str, go.Figure]:
    rice = df[df["rice_group"].notna()].copy()
    rice["commodity"] = "Rice (combined)"
    rice_monthly = rice.groupby(["commodity", "month"])["price"].mean().reset_index()

    trend = px.line(
        rice_monthly,
        x="month",
        y="price",
        color="commodity",
        template="plotly_white",
        color_discrete_sequence=[PALETTE[0]],
        title="Combined rice price trend",
    )
    trend = add_event_markers(style_fig(trend), active=True)
    trend.update_xaxes(tickangle=90)

    vol = rice.groupby("month")["price"].mean().pct_change().abs().reset_index(name="abs_change")
    volatility = px.line(
        vol,
        x="month",
        y="abs_change",
        color_discrete_sequence=[PALETTE[1]],
        template="plotly_white",
        title="Rice volatility (|pct change|)",
    )
    volatility.update_layout(legend=dict(x=1.08))
    volatility.update_xaxes(tickangle=90)

    inequality = (
        rice.groupby("admin1_label")["price"]
        .mean()
        .reset_index()
        .sort_values("price", ascending=False)
    )
    inequality_fig = px.bar(
        inequality,
        x="admin1_label",
        y="price",
        color="price",
        template="plotly_white",
        color_continuous_scale="YlGn",
        title="Regional inequality: rice price by state",
        labels={"admin1_label": "State", "price": "Avg rice price"},
    )
    inequality_fig.update_layout(xaxis_tickangle=90, showlegend=False, legend=dict(x=1.08))

    return {"trend": trend, "volatility": volatility, "inequality": inequality_fig}


def rice_affordability(df: pd.DataFrame) -> go.Figure:
    rice = df[df["commodity"] == "Rice (low quality)"]
    wage = df[df["commodity"] == "Wage (non-qualified labour, non-agricultural)"]
    rice_month = rice.groupby("month")["price"].mean()
    wage_month = wage.groupby("month")["price"].mean()
    idx = pd.concat([rice_month, wage_month], axis=1, join="inner").dropna()
    idx.columns = ["rice_price", "wage"]
    idx["affordability"] = idx["rice_price"] / idx["wage"]
    fig = px.line(
        idx.reset_index(),
        x="month",
        y="affordability",
        template="plotly_white",
        color_discrete_sequence=[PALETTE[1]],
        title="Rice affordability index (rice price / daily wage)",
        labels={"affordability": "Days of wage needed for 1 unit"},
    )
    fig.update_layout(legend=dict(x=1.08))
    fig.update_xaxes(tickangle=90)
    return fig


def mandalay_anomaly(df: pd.DataFrame) -> Tuple[Dict[str, float], go.Figure]:
    rice = df[(df["admin1_clean"] == "Mandalay") & (df["commodity"].isin(RICE_VARIANTS))]
    rice["period"] = np.where(rice["date"] < pd.Timestamp("2025-03-01"), "Pre-2025", "Post-2025")
    summary = rice.groupby("period")["price"].mean()
    national = (
        df[df["commodity"].isin(RICE_VARIANTS)]
        .groupby("month")["price"]
        .mean()
        .reset_index()
    )
    mandalay_monthly = rice.groupby("month")["price"].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=national["month"],
            y=national["price"],
            name="National rice avg",
            mode="lines",
            line=dict(color="#1f77b4"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=mandalay_monthly["month"],
            y=mandalay_monthly["price"],
            name="Mandalay rice",
            mode="lines",
            line=dict(color="#d62728"),
        )
    )
    fig = add_event_markers(fig, active=True)
    fig.update_layout(
        template="plotly_white",
        title="Mandalay vs national rice price with 2025 shock marker",
        hovermode="x unified",
        legend=dict(x=1.08),
    )
    fig.update_xaxes(tickangle=90)

    delta = (
        (summary.get("Post-2025", np.nan) - summary.get("Pre-2025", np.nan))
        / summary.get("Pre-2025", np.nan)
        * 100
    )
    shock_stats = {
        "pre_avg": float(summary.get("Pre-2025", np.nan)),
        "post_avg": float(summary.get("Post-2025", np.nan)),
        "pct_change": float(delta),
    }
    return shock_stats, fig


def rice_forecast(df: pd.DataFrame, periods: int = 12) -> go.Figure:
    series = (
        df[df["commodity"].isin(RICE_VARIANTS)]
        .groupby("month")["price"]
        .mean()
        .sort_index()
    )
    series = series.asfreq("MS").interpolate()
    model = ARIMA(series, order=(1, 1, 1))
    fit = model.fit()
    forecast = fit.get_forecast(steps=periods)
    pred = forecast.predicted_mean
    conf = forecast.conf_int(alpha=0.2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series, name="History", line=dict(color=PALETTE[0])))
    fig.add_trace(
        go.Scatter(
            x=pred.index,
            y=pred,
            name="Forecast",
            line=dict(color=PALETTE[1], dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pred.index.tolist() + pred.index[::-1].tolist(),
            y=conf["upper price"].tolist() + conf["lower price"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(102,187,106,0.15)",
            line=dict(color="rgba(102,187,106,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="80% CI",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title="Rice price forecast (ARIMA, 12-month horizon)",
        yaxis_title="Price (MMK)",
        legend=dict(x=1.08),
    )
    fig.update_xaxes(tickangle=30)
    return fig


def risk_scoring(df: pd.DataFrame) -> pd.DataFrame:
    recent = df[df["year"] >= 2021].copy()
    vol = (
        recent.groupby("admin1_clean")["price"]
        .agg(["mean", "std"])
        .assign(volatility=lambda x: x["std"] / x["mean"])
    )
    inflation = (
        recent.groupby(["admin1_clean", "year"])["price"]
        .mean()
        .groupby(level=0)
        .apply(lambda s: (s.iloc[-1] - s.iloc[0]) / s.iloc[0])
        .rename("inflation")
    )
    monthly = (
        recent.groupby(["admin1_clean", "commodity", "month"])["price"]
        .mean()
        .reset_index()
        .sort_values("month")
    )
    monthly["pct_change"] = monthly.groupby(["admin1_clean", "commodity"])["price"].pct_change()
    shocks = monthly[np.abs(monthly["pct_change"]) >= 0.2]
    shock_freq = shocks.groupby("admin1_clean").size().rename("shock_count")

    scores = vol.join(inflation).join(shock_freq).fillna(0)
    # Normalize each component for a simple composite score.
    for col in ["volatility", "inflation", "shock_count"]:
        col_min, col_max = scores[col].min(), scores[col].max()
        if col_max - col_min > 0:
            scores[col + "_norm"] = (scores[col] - col_min) / (col_max - col_min)
        else:
            scores[col + "_norm"] = 0
    scores["risk_score"] = (
        0.4 * scores["volatility_norm"]
        + 0.35 * scores["shock_count_norm"]
        + 0.25 * scores["inflation_norm"]
    )
    return scores.sort_values("risk_score", ascending=False).reset_index()


def narrative_block(title: str, bullet_points: List[str]) -> None:
    st.markdown(f"**{title}**")
    for b in bullet_points:
        st.write(f"- {b}")


def main() -> None:
    st.set_page_config(
        page_title="Myanmar Food Price Intelligence Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Myanmar Food Price Intelligence Dashboard (2008–2025)")
    st.caption("Food price dynamics, inequality, shocks, rice vulnerabilities, and policy relevance")

    df = load_data()
    hero = compute_hero_metrics(df)
    vol_by_year = (
        df.groupby("year")["price"].agg(["mean", "std"]).assign(cv=lambda x: x["std"] / x["mean"])
    ).sort_values("cv", ascending=False)
    volatility_peak_year = vol_by_year.index[0]
    total_regions = df["admin1_clean"].nunique()
    total_markets = df["market"].nunique()
    total_commodities = df["commodity"].nunique()

    st.sidebar.header("Navigation")
    sections = [
        "Overview",
        "National Trends",
        "Regional Story",
        "Commodity Deep Dive",
        "Rice Focus",
        "Shock & Volatility Frame",
        "Future Outlook & Risk",
        "Final Summary & Future Decisions",
    ]
    selected = st.sidebar.radio("Jump to section", sections)
    st.sidebar.markdown("Events overlay: 2015 Flood, 2020 COVID, 2021 Crisis, 2025 EQ.")
    st.sidebar.markdown("Thresholds: shocks = ±20% monthly change; volatility = CV.")

    st.sidebar.subheader("Hero metrics")
    st.sidebar.metric("Overall inflation", f"{hero.inflation_pct:,.0f}%")
    st.sidebar.metric("Shock events (|Δ|≥20%)", f"{hero.shock_events}")
    st.sidebar.metric("Highest volatility commodity", f"{hero.most_volatile} (CV {hero.most_volatile_cv:.2f})")
    st.sidebar.metric("Highest volatility period", f"{volatility_peak_year}")

    if selected == "Overview":
        st.header("Section 1 — Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall inflation (avg growth)", f"{hero.inflation_pct:,.1f}%")
            st.metric("Highest volatility period", f"{volatility_peak_year}")
        with col2:
            st.metric("Total regions", f"{total_regions}")
            st.metric("Total markets", f"{total_markets}")
        with col3:
            st.metric("Total commodities", f"{total_commodities}")
            st.metric("Most volatile commodity", f"{hero.most_volatile}", delta=f"CV {hero.most_volatile_cv:.2f}")

        st.write("Expanded view of the data structure and coverage across commodities, regions, and rice markets.")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(avg_price_by_category_fig(df), use_container_width=True)
            st.plotly_chart(top_commodities_recorded_fig(df), use_container_width=True)
        with c2:
            st.plotly_chart(avg_price_by_region_fig(df), use_container_width=True)
            st.plotly_chart(price_distribution_by_region_fig(df), use_container_width=True)
            st.caption("More developed regions like Yangon & Mandalay show tighter distributions, while conflict/underdeveloped regions show wider spreads.")

    if selected == "National Trends":
        st.header("Section 2 — National Trends")
        col1, col2 = st.columns([3, 2])
        with col1:
            st.plotly_chart(national_trend_plot(df, "MMK"), use_container_width=True)
            st.plotly_chart(yoy_change_plot(df), use_container_width=True)
        with col2:
            st.plotly_chart(commodity_index_plot(df, base_year=2010), use_container_width=True)
        st.info(
            "Myanmar’s food prices show a long-run upward movement with clear structural breaks in 2015, 2020, and especially in 2021 due to currency collapse and supply chain disruptions."
        )

    if selected == "Regional Story":
        st.header("Section 3 — Regional Story (Deep Dive)")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(pre_post_region_price_fig(df), use_container_width=True)
            st.plotly_chart(top_post_increase_regions_fig(df), use_container_width=True)
        with c2:
            st.plotly_chart(expensive_regions_fig(df), use_container_width=True)
            st.plotly_chart(regional_volatility_fig(df), use_container_width=True)
        st.plotly_chart(inequality_box_fig(df), use_container_width=True)
        st.caption("CV is calculated as standard deviation divided by mean price for each region.")
        st.info(
            "Regional inequality widens after 2021. Sagaing and northern conflict-affected regions show significantly elevated prices and volatility, indicating both insecurity and blocked trade routes."
        )

    if selected == "Commodity Deep Dive":
        st.header("Section 4 — Commodity Deep Dive")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(top_yearly_change_commodities_fig(df), use_container_width=True)
            st.plotly_chart(inflationary_since_2021_fig(df), use_container_width=True)
        with c2:
            most, stable = commodity_volatility(df)
            st.plotly_chart(most, use_container_width=True)
            st.plotly_chart(stable, use_container_width=True)
            cat_fig, _ = category_change(df)
            st.plotly_chart(cat_fig, use_container_width=True)
        st.info(
            "Oils, proteins, and perishable vegetables lead volatility and inflation. Staples like salt and Emata rice show relative stability."
        )

    if selected == "Rice Focus":
        st.header("Section 5 — Rice Focus")
        figs = rice_section(df)
        st.plotly_chart(figs["trend"], use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(figs["volatility"], use_container_width=True)
            st.plotly_chart(rice_affordability(df), use_container_width=True)
        with col2:
            st.plotly_chart(figs["inequality"], use_container_width=True)
            stats, mandalay_fig = mandalay_anomaly(df)
            st.plotly_chart(mandalay_fig, use_container_width=True)
            st.metric("Mandalay 2025 shock Δ", f"{stats['pct_change']:.1f} %", help="Post-2025 vs pre-2025 average rice price")
        st.info(
            "As Myanmar’s essential staple, rice prices are critical for welfare. The 2025 Mandalay earthquake triggered sharp isolated increases, reflecting the region’s reliance on central transport corridors."
        )

    if selected == "Shock & Volatility Frame":
        st.header("Section 6 — Shock & Volatility Frame")
        shocks, heatmap = shocks_by_state(df)
        c1, c2 = st.columns(2)
        with c1:
            vuln = shocks.groupby("admin1_clean").size().sort_values(ascending=False).reset_index(name="shock_count")
            vuln["admin1_label"] = vuln["admin1_clean"].apply(lambda x: shorten_label(x, 18))
            fig_vuln = px.bar(
                vuln.head(10),
                x="admin1_label",
                y="shock_count",
                template="plotly_white",
                color="shock_count",
                color_continuous_scale="YlGn",
                title="Top 10 shock-vulnerable states (|Δmonth|≥20%)",
                labels={"admin1_label": "State", "shock_count": "Shock count"},
            )
            fig_vuln.update_layout(showlegend=False, legend=dict(x=1.08))
            fig_vuln.update_xaxes(tickangle=30)
            st.plotly_chart(fig_vuln, use_container_width=True)
        with c2:
            most, stable = commodity_volatility(df)
            st.plotly_chart(most, use_container_width=True)
            st.plotly_chart(stable, use_container_width=True)
        st.info("Shocks cluster heavily around national disruptions: 2015 floods, COVID-19, and the 2021 political crisis.")

    if selected == "Future Outlook & Risk":
        st.header("Section 7 — Future Outlook & Risk Assessment")
        st.plotly_chart(rice_forecast(df), use_container_width=True)
        scores = risk_scoring(df)
        scores["admin1_label"] = scores["admin1_clean"].apply(lambda x: shorten_label(x, 18))
        fig_risk = px.bar(
            scores.head(15),
            x="admin1_label",
            y="risk_score",
            template="plotly_white",
            color="risk_score",
            color_continuous_scale="YlGn",
            title="State risk scores (volatility_norm + shock_count_norm + inflation_norm)",
            labels={"admin1_label": "State", "risk_score": "Composite risk"},
        )
        fig_risk.update_layout(showlegend=False, legend=dict(x=1.08))
        fig_risk.update_xaxes(tickangle=30)
        st.plotly_chart(fig_risk, use_container_width=True)
        st.info(
            "Risk modeling suggests rising vulnerability in central and northern states, demanding early-warning systems and supply-chain resilience measures."
        )

    if selected == "Final Summary & Future Decisions":
        st.header("Section 8 — Final Summary & Future Decisions")
        st.markdown("**FINAL SUMMARY**")
        st.markdown(
            """
- Myanmar’s food price system demonstrates long-run structural inflation, amplified after 2021.
- Regional inequality has increased, with conflict-affected regions (Sagaing, Kachin, Chin) facing the highest surges.
- Rice remains the anchor of household consumption, and its rising affordability gap is a major welfare concern.
- Shock patterns follow major national crises (2015 flood, 2020 COVID, 2021 coup), showing market fragility.
- Volatility has increased for oils, pulses, and perishable vegetables, indicating supply chain bottlenecks.
"""
        )
        st.markdown("**FUTURE DECISION PATH (Actionable recommendations)**")
        st.markdown(
            """
- Strengthen logistics in high-risk regions (Sagaing, Mandalay, Kachin)
- Pre-position essential food stocks in shock-prone areas
- Build rice market buffers and stabilization programs
- Introduce real-time monitoring systems using monthly shock detection
- Targeted cash/wage support for price spikes
- Develop local storage/processing capacity to reduce spoilage-driven volatility
- Use rice affordability index as an early-warning indicator
- Incorporate climate data for flood- and drought-based forecasting
"""
        )
        st.success(
            "This dashboard supports data-driven humanitarian planning, economic monitoring, and strategic market interventions for Myanmar’s food security landscape."
        )


if __name__ == "__main__":
    main()
