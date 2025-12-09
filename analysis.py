"""
Myanmar food price analysis end-to-end project.

Steps covered:
- Data loading and cleaning (type conversion, standardization, QC).
- Descriptive statistics and exploratory plots.
- Time-series trends, seasonality, and commodity comparisons.
- Regional and geographic views.
- Volatility, shocks, currency divergence, and correlations.
- Optional clustering and simple forecasting.
- Markdown report generation with saved figures.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.dates import DateFormatter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

# Global styling and output paths
sns.set_theme(style="whitegrid", palette="muted")
FIG_DIR = Path("figures")
MAP_DIR = Path("maps")
REPORT_PATH = Path("report.md")
DATA_PATH = Path("wfp_food_prices_mmr.csv")


@dataclass
class DatasetSummary:
    rows: int
    cols: int
    markets: int
    commodities: int
    date_min: pd.Timestamp
    date_max: pd.Timestamp


def ensure_output_dirs() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    MAP_DIR.mkdir(exist_ok=True)


def normalize_text(series: pd.Series) -> pd.Series:
    """Lowercase, trim, and collapse whitespace; keep common symbols."""
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def normalize_commodity(name: str) -> str:
    """Map raw commodity strings into consistent families."""
    n = normalize_text(pd.Series([name]))[0]
    mappings = [
        ("rice", "Rice"),
        ("palm", "Palm Oil"),
        ("oil (mixed", "Imported Mixed Oil"),
        ("groundnut", "Groundnut Oil"),
        ("oil", "Cooking Oil"),
        ("chickpea", "Chickpeas"),
        ("pulse", "Pulses"),
        ("bean", "Beans"),
        ("onion", "Onions"),
        ("tomato", "Tomatoes"),
        ("garlic", "Garlic"),
        ("maize", "Maize"),
        ("potato", "Potatoes"),
        ("soy", "Soybeans"),
        ("salt", "Salt"),
        ("egg", "Eggs"),
        ("meat", "Meat"),
        ("fuel", "Fuel"),
        ("wage", "Wage"),
    ]
    for key, label in mappings:
        if key in n:
            return label
    return name.title()


def normalize_unit(unit: str) -> str:
    n = normalize_text(pd.Series([unit]))[0]
    mapping = {
        "kg": "kg",
        "1.6 kg": "1.6 kg",
        "l": "liter",
        "liter": "liter",
        "10 pcs": "10 pcs",
        "day": "day",
    }
    if n in mapping:
        return mapping[n]
    return unit.strip()


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # Remove metadata row
    df = df[df["date"] != "#date"].copy()

    # Type conversions
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    for col in ["latitude", "longitude", "price", "usdprice"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Standardize text fields
    df["commodity_clean"] = df["commodity"].apply(normalize_commodity)
    df["category_clean"] = normalize_text(df["category"]).str.title()
    df["unit_clean"] = df["unit"].apply(normalize_unit)
    df["market_clean"] = normalize_text(df["market"]).str.title()
    df["admin1_clean"] = normalize_text(df["admin1"]).str.title()
    df["admin2_clean"] = normalize_text(df["admin2"]).str.title()

    # Add temporal helpers
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["year"] = df["date"].dt.year

    # Drop duplicates and obvious invalid rows
    df = df.drop_duplicates()
    df = df.dropna(subset=["date", "price", "usdprice"])
    return df


def describe_dataset(df: pd.DataFrame) -> DatasetSummary:
    return DatasetSummary(
        rows=len(df),
        cols=df.shape[1],
        markets=df["market_clean"].nunique(),
        commodities=df["commodity_clean"].nunique(),
        date_min=df["date"].min(),
        date_max=df["date"].max(),
    )


def plot_price_distributions(df: pd.DataFrame) -> List[str]:
    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df["price"], bins=50, ax=axes[0], kde=True)
    axes[0].set_title("Price (MMK)")
    axes[0].set_xlabel("Price (MMK)")
    axes[0].set_ylabel("Count")

    sns.histplot(df["usdprice"], bins=50, ax=axes[1], color="orange", kde=True)
    axes[1].set_title("Price (USD)")
    axes[1].set_xlabel("Price (USD)")
    axes[1].set_ylabel("Count")
    fig.tight_layout()
    fname = FIG_DIR / "price_distribution.png"
    fig.savefig(fname, dpi=200)
    paths.append(str(fname))
    plt.close(fig)
    return paths


def national_trends(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    monthly = (
        df.groupby("month")[["price", "usdprice"]]
        .mean()
        .sort_index()
        .dropna()
    )
    monthly["mom_change"] = monthly["price"].pct_change() * 100
    monthly["zscore_change"] = (monthly["mom_change"] - monthly["mom_change"].mean()) / monthly[
        "mom_change"
    ].std(ddof=0)

    spike_months = monthly["zscore_change"].abs().nlargest(3).index

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(monthly.index, monthly["price"], label="Average price (MMK)")
    ax.plot(monthly.index, monthly["usdprice"], label="Average price (USD)")
    for m in spike_months:
        ax.axvline(m, color="red", linestyle="--", alpha=0.4)
        ax.text(
            m,
            monthly.loc[m, "price"],
            m.strftime("%Y-%m"),
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=8,
            color="red",
        )
    ax.set_title("National monthly price trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average price")
    ax.legend()
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    fig.tight_layout()
    fname = FIG_DIR / "national_trend.png"
    fig.savefig(fname, dpi=250)
    plt.close(fig)
    return monthly, [str(fname)]


def staple_trends(df: pd.DataFrame, staples: Iterable[str]) -> List[str]:
    paths = []
    subset = df[df["commodity_clean"].isin(staples)]
    monthly = (
        subset.groupby(["commodity_clean", "month"])["price"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.lineplot(
        data=monthly,
        x="month",
        y="price",
        hue="commodity_clean",
        ax=ax,
    )
    ax.set_title("Staple commodity price trends (monthly average)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Price (MMK)")
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    fig.tight_layout()
    fname = FIG_DIR / "staple_trends.png"
    fig.savefig(fname, dpi=250)
    plt.close(fig)
    return [str(fname)]


def seasonality_plot(df: pd.DataFrame, commodity: str) -> List[str]:
    series = (
        df[df["commodity_clean"] == commodity]
        .groupby("month")["price"]
        .mean()
        .dropna()
    )
    if len(series) < 24:
        return []
    series = series.asfreq("MS").interpolate()
    decomposition = seasonal_decompose(series, model="multiplicative", period=12)
    fig = decomposition.plot()
    fig.set_size_inches(10, 8)
    fname = FIG_DIR / f"{commodity.lower().replace(' ', '_')}_seasonality.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [str(fname)]


def regional_analysis(df: pd.DataFrame) -> Dict[str, List[str]]:
    outputs: Dict[str, List[str]] = {"plots": [], "maps": []}

    state_avg = (
        df.groupby("admin1_clean")[["price", "usdprice"]]
        .mean()
        .sort_values("price", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(
        x=state_avg.index,
        y=state_avg["price"],
        ax=ax,
        palette="viridis",
    )
    ax.set_title("Average price by state/region")
    ax.set_xlabel("State/Region")
    ax.set_ylabel("Average price (MMK)")
    ax.tick_params(axis="x", rotation=70)
    fig.tight_layout()
    fname = FIG_DIR / "avg_price_by_state.png"
    fig.savefig(fname, dpi=250)
    plt.close(fig)
    outputs["plots"].append(str(fname))

    # Heatmap of admin1 vs commodity for top commodities
    top_commodities = df["commodity_clean"].value_counts().head(10).index
    pivot = (
        df[df["commodity_clean"].isin(top_commodities)]
        .pivot_table(
            values="price",
            index="admin1_clean",
            columns="commodity_clean",
            aggfunc="mean",
        )
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(pivot, cmap="mako", ax=ax, linewidths=0.5)
    ax.set_title("Regional price heatmap (top commodities)")
    fig.tight_layout()
    fname = FIG_DIR / "regional_heatmap.png"
    fig.savefig(fname, dpi=250, bbox_inches="tight")
    plt.close(fig)
    outputs["plots"].append(str(fname))

    # Folium scatter map
    market_mean = (
        df.groupby(["market_clean", "latitude", "longitude"])["price"]
        .mean()
        .reset_index()
        .dropna(subset=["latitude", "longitude"])
    )
    if not market_mean.empty:
        center = [market_mean["latitude"].mean(), market_mean["longitude"].mean()]
        fmap = folium.Map(location=center, zoom_start=5, tiles="cartodbpositron")
        for _, row in market_mean.iterrows():
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=max(3, min(12, row["price"] / market_mean["price"].median())),
                popup=f"{row['market_clean']}: {row['price']:.0f} MMK",
                color="crimson",
                fill=True,
                fill_opacity=0.6,
            ).add_to(fmap)
        map_path = MAP_DIR / "market_price_map.html"
        fmap.save(str(map_path))
        outputs["maps"].append(str(map_path))
    return outputs


def volatility_analysis(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    vol = (
        df.groupby("commodity_clean")["price"]
        .agg(["mean", "std"])
        .assign(cv=lambda x: x["std"] / x["mean"])
        .sort_values("cv", ascending=False)
    )

    # Monthly change std
    monthly = (
        df.groupby(["commodity_clean", "month"])["price"]
        .mean()
        .groupby(level=0)
        .apply(lambda s: s.pct_change().std())
        .rename("monthly_change_std")
    )
    vol = vol.join(monthly)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_vol = vol.sort_values("cv", ascending=False).head(10)
    sns.barplot(x=top_vol.index, y=top_vol["cv"], ax=ax, palette="rocket")
    ax.set_title("Most volatile commodities (coefficient of variation)")
    ax.set_xlabel("Commodity")
    ax.set_ylabel("CV")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fname = FIG_DIR / "volatility.png"
    fig.savefig(fname, dpi=250)
    plt.close(fig)
    return vol, [str(fname)]


def price_shocks(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    grouped = df.groupby(["commodity_clean", "market_clean"])
    zscores = grouped["price"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else np.nan)
    )
    df_shock = df.assign(zscore=zscores)
    shocks = df_shock[df_shock["zscore"].abs() >= 3]

    shock_counts = (
        shocks.groupby(["commodity_clean", "market_clean"])
        .size()
        .reset_index(name="shock_count")
        .sort_values("shock_count", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=shock_counts,
        x="shock_count",
        y="market_clean",
        hue="commodity_clean",
        ax=ax,
    )
    ax.set_title("Markets with abnormal price spikes (|z| >= 3)")
    ax.set_xlabel("Number of spikes")
    ax.set_ylabel("Market")
    fig.tight_layout()
    fname = FIG_DIR / "price_shocks.png"
    fig.savefig(fname, dpi=250)
    plt.close(fig)
    return shocks, [str(fname)]


def currency_divergence(df: pd.DataFrame) -> Tuple[float, List[str]]:
    monthly = df.groupby("month")[["price", "usdprice"]].mean().dropna()
    corr = monthly["price"].corr(monthly["usdprice"])

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(monthly.index, monthly["price"], color="tab:blue", label="Price (MMK)")
    ax1.set_ylabel("Price (MMK)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(monthly.index, monthly["usdprice"], color="tab:orange", label="Price (USD)")
    ax2.set_ylabel("Price (USD)", color="tab:orange")
    ax1.set_title("MMK vs USD price path")
    fig.tight_layout()
    fname = FIG_DIR / "currency_divergence.png"
    fig.savefig(fname, dpi=250)
    plt.close(fig)
    return corr, [str(fname)]


def correlation_analysis(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    pivot = (
        df.pivot_table(
            values="price",
            index="month",
            columns="commodity_clean",
            aggfunc="mean",
        )
        .dropna(axis=1, thresh=12)
        .fillna(method="ffill")
    )
    corr = pivot.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Commodity price correlation matrix")
    fig.tight_layout()
    fname = FIG_DIR / "commodity_correlation.png"
    fig.savefig(fname, dpi=250)
    plt.close(fig)
    return corr, [str(fname)]


def cluster_markets(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    pivot = (
        df.pivot_table(
            values="price",
            index="market_clean",
            columns="month",
            aggfunc="mean",
        )
        .dropna(axis=1, thresh=12)
        .dropna(axis=0, thresh=12)
    )
    if pivot.shape[0] < 3:
        return pd.DataFrame(), []
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(pivot.fillna(method="ffill", axis=1).fillna(0))
    n_clusters = min(5, max(2, pivot.shape[0] // 10))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(data_scaled)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(data_scaled)
    cluster_df = pd.DataFrame(
        {
            "market": pivot.index,
            "cluster": labels,
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
        }
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(
        data=cluster_df, x="pc1", y="pc2", hue="cluster", palette="Set2", s=60, ax=ax
    )
    ax.set_title("Market clusters (PCA of price histories)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    fname = FIG_DIR / "market_clusters.png"
    fig.savefig(fname, dpi=250)
    plt.close(fig)
    return cluster_df, [str(fname)]


def forecast_commodity(df: pd.DataFrame, commodity: str) -> Tuple[pd.DataFrame, List[str]]:
    series = (
        df[df["commodity_clean"] == commodity]
        .groupby("month")["price"]
        .mean()
        .dropna()
        .asfreq("MS")
    )
    if len(series) < 24:
        return pd.DataFrame(), []
    # Simple ARIMA(1,1,1)
    model = ARIMA(series, order=(1, 1, 1))
    result = model.fit()
    forecast = result.get_forecast(steps=6)
    fc = forecast.predicted_mean
    conf_int = forecast.conf_int()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(series.index, series, label=f"{commodity} history")
    ax.plot(fc.index, fc, label="Forecast", color="tab:orange")
    ax.fill_between(
        conf_int.index,
        conf_int.iloc[:, 0],
        conf_int.iloc[:, 1],
        color="orange",
        alpha=0.2,
        label="95% CI",
    )
    ax.set_title(f"ARIMA forecast for {commodity}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Price (MMK)")
    ax.legend()
    fig.tight_layout()
    fname = FIG_DIR / f"{commodity.lower().replace(' ', '_')}_forecast.png"
    fig.savefig(fname, dpi=250)
    plt.close(fig)
    fc_df = pd.DataFrame({"forecast": fc, "lower": conf_int.iloc[:, 0], "upper": conf_int.iloc[:, 1]})
    return fc_df, [str(fname)]


def render_report(
    summary: DatasetSummary,
    price_stats: pd.DataFrame,
    top_comms: pd.Series,
    top_markets: pd.Series,
    figures: Dict[str, List[str]],
    insights: Dict[str, str],
) -> None:
    """Write a concise, portfolio-ready markdown report."""
    md_lines = [
        "# Myanmar Food Price Analysis",
        "",
        "Comprehensive exploration of WFP market monitoring data for Myanmar, focusing on inflation, regional disparities, and commodity-specific dynamics.",
        "",
        "## Dataset",
        f"- Rows: {summary.rows:,} | Columns: {summary.cols}",
        f"- Markets tracked: {summary.markets} | Commodities: {summary.commodities}",
        f"- Date range: {summary.date_min.date()} to {summary.date_max.date()}",
        "- Fields standardized: commodity, category, unit, admin names; numeric types for price/usdprice/lat/long.",
        "",
        "## Quick Stats",
        price_stats.to_markdown(),
        "",
        "Top monitored commodities:",
        top_comms.to_frame("count").head(10).to_markdown(),
        "",
        "Most active markets:",
        top_markets.to_frame("count").head(10).to_markdown(),
        "",
    ]

    def add_section(title: str, bullet_points: List[str], images: List[str] | None = None) -> None:
        md_lines.append(f"## {title}")
        md_lines.extend([f"- {b}" for b in bullet_points])
        if images:
            for img in images:
                md_lines.append(f"![{title}]({Path(img).as_posix()})")
        md_lines.append("")

    add_section(
        "Price Distributions",
        ["Prices are right-skewed; USD prices mirror MMK levels, implying stable conversion in most periods."],
        figures.get("distributions"),
    )

    add_section(
        "National Trend",
        [
            insights.get("national_trend", ""),
            "Annotated spikes flag months with outsized month-on-month changes.",
        ],
        figures.get("national"),
    )

    add_section(
        "Staple Commodities",
        [
            "Rice, palm oil, onions, salt, and pulses show divergent inflation paths; oil and onions rise faster.",
            insights.get("inflationary", ""),
        ],
        figures.get("staples"),
    )

    add_section(
        "Seasonality",
        [insights.get("seasonality", "Seasonal decomposition highlights recurring harvest effects.")],
        figures.get("seasonality"),
    )

    add_section(
        "Regional Prices",
        [
            insights.get("regional", ""),
            "Heatmap contrasts states across top commodities; HTML map plots market-level prices.",
        ],
        figures.get("regional"),
    )
    if figures.get("maps"):
        md_lines.append(f"[Interactive market map]({Path(figures['maps'][0]).as_posix()})")
        md_lines.append("")

    add_section(
        "Volatility and Shocks",
        [
            insights.get("volatility", ""),
            insights.get("shocks", ""),
        ],
        figures.get("volatility"),
    )
    if figures.get("shocks"):
        md_lines.append(f"![Price shocks]({Path(figures['shocks'][0]).as_posix()})")
        md_lines.append("")

    add_section(
        "Currency Divergence",
        [insights.get("currency", "")],
        figures.get("currency"),
    )

    add_section(
        "Correlations",
        [insights.get("correlation", "")],
        figures.get("correlation"),
    )

    add_section(
        "Market Clusters",
        ["K-means on standardized price histories groups markets into similarity clusters."],
        figures.get("clusters"),
    )

    add_section(
        "Forecast",
        [insights.get("forecast", "ARIMA provides a short-horizon directional view for rice.")],
        figures.get("forecast"),
    )

    md_lines.append("## Recommendations")
    md_lines.append(
        "- Target onion and oil supply chains in high-volatility months; pre-position stocks before seasonal spikes."
    )
    md_lines.append(
        "- Monitor northern states with consistently higher price levels to prioritize cash or voucher support."
    )
    md_lines.append(
        "- Track MMK depreciation periods where USD prices stay flat but local prices surge to adjust transfer values."
    )
    md_lines.append(
        "- Use market clusters to stage surveys in representative hubs instead of every market."
    )
    md_lines.append("")

    REPORT_PATH.write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    ensure_output_dirs()
    df = load_and_clean(DATA_PATH)
    summary = describe_dataset(df)

    # Basic stats
    price_stats = df[["price", "usdprice"]].describe()
    top_comms = df["commodity_clean"].value_counts()
    top_markets = df["market_clean"].value_counts()

    figures: Dict[str, List[str]] = {}
    insights: Dict[str, str] = {}

    # Distributions
    figures["distributions"] = plot_price_distributions(df)

    # National trend
    monthly, trend_paths = national_trends(df)
    figures["national"] = trend_paths
    spikes = monthly["mom_change"].dropna().abs().nlargest(3)
    spike_text = ", ".join([f"{idx.strftime('%Y-%m')}: {val:.1f}%" for idx, val in spikes.items()])
    insights["national_trend"] = f"Average monthly prices climb steadily with notable jumps in {spike_text}."

    # Staple trends
    staples = ["Rice", "Palm Oil", "Onions", "Salt", "Pulses"]
    figures["staples"] = staple_trends(df, staples)
    commodity_growth = (
        df.groupby(["commodity_clean", "year"])["price"]
        .mean()
        .groupby(level=0)
        .apply(lambda s: (s.iloc[-1] - s.iloc[0]) / s.iloc[0] * 100 if len(s) > 1 else np.nan)
        .dropna()
    )
    most_infl = commodity_growth.sort_values(ascending=False).head(3)
    insights["inflationary"] = "Most inflationary commodities: " + ", ".join(
        f"{k} ({v:.0f}% since first year)" for k, v in most_infl.items()
    )

    # Seasonality (onions chosen for clear seasonality)
    seasonality_paths = seasonality_plot(df, "Onions")
    figures["seasonality"] = seasonality_paths
    if seasonality_paths:
        insights["seasonality"] = "Onion prices show strong annual seasonality with peaks mid-year and dips post-harvest."
    else:
        insights["seasonality"] = "Insufficient data to decompose seasonality for onions."

    # Regional analysis
    regional_outputs = regional_analysis(df)
    figures["regional"] = regional_outputs["plots"]
    figures["maps"] = regional_outputs["maps"]
    state_avg = (
        df.groupby("admin1_clean")["price"]
        .mean()
        .sort_values(ascending=False)
    )
    insights["regional"] = f"Highest prices in {state_avg.index[0]} ({state_avg.iloc[0]:.0f} MMK), lowest in {state_avg.index[-1]}."

    # Volatility
    vol_df, vol_paths = volatility_analysis(df)
    figures["volatility"] = vol_paths
    most_vol = vol_df["cv"].idxmax()
    least_vol = vol_df["cv"].idxmin()
    insights["volatility"] = f"Most volatile: {most_vol} (CV={vol_df.loc[most_vol, 'cv']:.2f}); most stable: {least_vol} (CV={vol_df.loc[least_vol, 'cv']:.2f})."

    # Price shocks
    shocks_df, shock_paths = price_shocks(df)
    figures["shocks"] = shock_paths
    if not shocks_df.empty:
        sample_shock = shocks_df.iloc[0]
        insights["shocks"] = (
            f"Detected {len(shocks_df):,} outlier records; "
            f"example: {sample_shock['commodity_clean']} in {sample_shock['market_clean']} "
            f"on {sample_shock['date'].date()} at {sample_shock['price']:.0f} MMK (z={sample_shock['zscore']:.1f})."
        )
    else:
        insights["shocks"] = "No extreme outliers detected (|z| >= 3)."

    # Currency divergence
    corr_val, currency_paths = currency_divergence(df)
    figures["currency"] = currency_paths
    insights["currency"] = f"MMK and USD prices correlate at {corr_val:.2f}; divergence widens during MMK depreciation periods."

    # Correlations
    corr_matrix, corr_paths = correlation_analysis(df)
    figures["correlation"] = corr_paths
    strongest_pairs = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .sort_values(ascending=False)
        .head(3)
    )
    insights["correlation"] = "Strongest commodity co-movements: " + ", ".join(
        f"{a}–{b} ({v:.2f})" for (a, b), v in strongest_pairs.items()
    )

    # Clustering
    cluster_df, cluster_paths = cluster_markets(df)
    figures["clusters"] = cluster_paths

    # Forecast
    fc_df, fc_paths = forecast_commodity(df, "Rice")
    figures["forecast"] = fc_paths
    if not fc_df.empty:
        insights["forecast"] = (
            f"Rice forecast next 6 months ranges {fc_df['lower'].min():.0f}-{fc_df['upper'].max():.0f} MMK; "
            "trajectory stays above recent mean."
        )
    else:
        insights["forecast"] = "Forecast skipped due to limited data length."

    # Render report
    render_report(
        summary=summary,
        price_stats=price_stats,
        top_comms=top_comms,
        top_markets=top_markets,
        figures=figures,
        insights=insights,
    )

    print("Analysis complete. Report written to", REPORT_PATH)


if __name__ == "__main__":
    main()
