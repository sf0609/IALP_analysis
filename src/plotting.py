import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def highlight_min(s):
    is_min = s == s.min()
    return ["background-color: lightgreen" if cell else "" for cell in is_min]


def highlight_max(s):
    is_max = s == s.max()
    return ["background-color: tomato" if cell else "" for cell in is_max]


def plot_examples():
    """Plot examples of time series"""
    csvs = ["sunspots", "co2", "bitcoin", "airline"]
    titles = [
        "Annual sunspots",
        "Monthly mean atmospheric C02",
        "Daily Bitcoin price",
        "Monthly total airline passengers",
    ]
    ytitles = ["Sunspots", "CO2 (ppm)", "$ (USD)", "Passengers"]
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=titles,
        horizontal_spacing=0.1,
        vertical_spacing=0.12,
    )
    for i, csv in enumerate(csvs):
        df = pd.read_csv(f"data/{csv}.csv", index_col=0, parse_dates=True)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df.iloc[:, 0],
                mode="lines",
                line=dict(width=2),
                name=csv,
            ),
            row=i // 2 + 1,
            col=i % 2 + 1,
        )
        fig.update_yaxes(
            title=ytitles[i], title_standoff=0, row=i // 2 + 1, col=i % 2 + 1
        )
        fig.update_xaxes(
            title="Time", title_standoff=0, row=i // 2 + 1, col=i % 2 + 1
        )
    fig.update_layout(
        width=800,
        height=600,
        title_x=0.5,
        title_y=0.93,
        margin=dict(t=60, b=10),
        showlegend=False,
    )
    return fig


def plot_stationary():
    """Plot examples of time series"""
    csvs = ["co2", "sunspots", "white-noise", "soi"]
    titles = [
        "Monthly mean atmospheric C02",
        "Annual sunspots",
        "White noise",
        "Southern Oscillation Index",
    ]
    ytitles = ["Sunspots", "CO2 (ppm)", "White noise", "SOI"]
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=titles,
        horizontal_spacing=0.1,
        vertical_spacing=0.12,
    )
    for i, csv in enumerate(csvs):
        df = pd.read_csv(f"data/{csv}.csv", index_col=0, parse_dates=True)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df.iloc[:, 0],
                mode="lines",
                line=dict(width=2),
                name=csv,
            ),
            row=i // 2 + 1,
            col=i % 2 + 1,
        )
        fig.update_yaxes(
            title=ytitles[i], title_standoff=0, row=i // 2 + 1, col=i % 2 + 1
        )
        fig.update_xaxes(
            title="Time", title_standoff=0, row=i // 2 + 1, col=i % 2 + 1
        )
    fig.update_layout(
        width=800,
        height=600,
        title_x=0.5,
        title_y=0.93,
        margin=dict(t=60, b=10),
        showlegend=False,
    )
    return fig


def plot_add_mult():
    """Plot examples of time series"""
    csvs = ["co2", "airline"]
    titles = [
        "Atmospheric C02 (additive)",
        "Airline passengers (multiplicative)",
    ]
    ytitles = ["CO2 (ppm)", "Passengers"]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=titles,
        horizontal_spacing=0.1,
        vertical_spacing=0.12,
    )
    for i, csv in enumerate(csvs):
        df = pd.read_csv(f"data/{csv}.csv", index_col=0, parse_dates=True)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df.iloc[:, 0],
                mode="lines",
                line=dict(width=2),
                name=csv,
            ),
            row=i // 2 + 1,
            col=i % 2 + 1,
        )
        fig.update_yaxes(
            title=ytitles[i], title_standoff=0, row=i // 2 + 1, col=i % 2 + 1
        )
        fig.update_xaxes(
            title="Time", title_standoff=0, row=i // 2 + 1, col=i % 2 + 1
        )
    fig.update_layout(
        width=900,
        height=350,
        title_x=0.5,
        title_y=0.93,
        margin=dict(t=60, b=10),
        showlegend=False,
    )
    return fig


def plot_autocorrelation(
    df, label, dlag=10, max_lag=None, axis_limits=[0, 200]
):
    """Interactive plot of autocorrelations at different lags"""
    y = df[label]
    if max_lag is None:
        max_lag = len(y)
    elif max_lag > len(y):
        max_lag = len(y)
    fig = go.Figure()
    p = (
        sm.OLS(y, sm.add_constant(y.shift(1)), missing="drop")
        .fit()
        .predict(sm.add_constant(y.shift(1).sort_values()))
    )
    fig.add_trace(
        go.Scatter(
            x=y.shift(1), y=y, mode="markers", marker=dict(size=8), name="Data"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=y.shift(1).sort_values(),
            y=p,
            mode="lines",
            line=dict(width=3),
            name="OLS",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[34],
            y=[186],
            mode="text",
            text=f"<i>r = {y.autocorr(-1):.2f}<i>",
            textfont=dict(size=18, color="red"),
            showlegend=False,
        )
    )
    fig.update_xaxes(
        range=axis_limits,
        tick0=0,
        dtick=25,
        title=f"Lagged {label}",
        title_standoff=0,
    )
    fig.update_yaxes(
        range=axis_limits, tick0=0, dtick=25, title=label, title_standoff=0
    )
    fig.update_layout(
        width=450,
        height=450,
        title="Autocorrelation",
        title_x=0.5,
        title_y=0.93,
        margin=dict(t=60),
    )

    frames = [
        dict(
            name=lag,
            data=[
                go.Scatter(x=y.shift(lag), y=y),
                go.Scatter(
                    x=y.shift(lag).sort_values(),
                    y=sm.OLS(y, sm.add_constant(y.shift(lag)), missing="drop")
                    .fit()
                    .predict(sm.add_constant(y.shift(lag).sort_values())),
                ),
                go.Scatter(text=f"<i>r = {y.autocorr(lag):.2f}<i>"),
            ],
            traces=[0, 1, 2],
        )
        for lag in range(1, max_lag, dlag)
    ]

    sliders = [
        {
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "Lag: ",
                "visible": True,
            },
            "pad": {"b": 10, "t": 30},
            "steps": [
                {
                    "args": [
                        [lag],
                        {
                            "frame": {
                                "duration": 0,
                                "easing": "linear",
                                "redraw": False,
                            },
                            "transition": {"duration": 0, "easing": "linear"},
                        },
                    ],
                    "label": lag,
                    "method": "animate",
                }
                for lag in range(1, max_lag, dlag)
            ],
        }
    ]
    fig.update(frames=frames), fig.update_layout(sliders=sliders)
    return fig


def plot_prediction_intervals(
    y,
    p,
    col="mean",
    coverage="95%",
    valid=None,
    xlabel=None,
    ylabel=None,
    width=700,
    height=400,
):
    """
    y = series
    p = prediction dataframe from statsmodels .summary_frame() method
    """
    if xlabel is None:
        xlabel = y.index.name

    if ylabel is None:
        ylabel = y.name

    if "pi_lower" not in p.columns:
        pilabel = "mean_ci"
    else:
        pilabel = "pi"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y.index, y=y, mode="lines", name="Observed"))
    if valid is not None:
        fig.add_trace(
            go.Scatter(
                x=valid.index,
                y=valid,
                mode="lines",
                line=dict(color="#00CC96"),
                name="Validation",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=p.index,
            y=p[col],
            mode="lines",
            line=dict(color="salmon"),
            name=f"{col.title()} predicted",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=p.index,
            y=p[f"{pilabel}_lower"],
            mode="lines",
            line=dict(width=0.5, color="salmon"),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=p.index,
            y=p[f"{pilabel}_upper"],
            mode="lines",
            line=dict(width=0.5, color="salmon"),
            fillcolor="rgba(250, 128, 114, 0.2)",
            fill="tonexty",
            name=f"{coverage} prediction interval",
        )
    )
    fig.update_xaxes(title=xlabel, title_standoff=0)
    fig.update_yaxes(title=ylabel, title_standoff=0)
    fig.update_layout(
        width=width,
        height=height,
        title_x=0.5,
        title_y=0.93,
        margin=dict(t=60, b=10),
    )
    return fig


def plot_interactive_differencing(y, seasonal=12, width=700, height=500):
    col = y.name
    df = pd.DataFrame(
        {
            col: y,
            f"{col}: diffx1": y.diff(),
            f"{col}: diffx2": y.diff().diff(),
            f"{col}: seasonal-diffx{seasonal}": y.diff(seasonal),
            f"{col}: seasonal-diffx{seasonal} + diffx1": y.diff(
                seasonal
            ).diff(),
        }
    )

    fig = go.Figure()
    cols = df.columns.to_list()
    colours = [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
    ]

    for i, column in enumerate(cols):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[column],
                name=column,
                line=dict(color=colours[i]),
            )
        )

    button_all = [
        dict(
            label="All",
            method="update",
            args=[{"visible": [True] * len(cols)}, {"showlegend": True}],
        )
    ]
    button_all_but_col = [
        dict(
            label="All differenced",
            method="update",
            args=[
                {"visible": [False] + [True] * (len(cols) - 1)},
                {"showlegend": True},
            ],
        )
    ]
    buttons = [
        dict(
            label=col,
            method="update",
            args=[
                {"visible": [True if col == _ else False for _ in cols]},
                {"showlegend": True},
            ],
        )
        for col in cols
    ]
    fig.update_xaxes(title=f"Time", title_standoff=0)
    fig.update_yaxes(title=col, title_standoff=0)
    fig.update_layout(
        updatemenus=[
            go.layout.Updatemenu(
                active=0,
                buttons=button_all + button_all_but_col + buttons,
                direction="down",
                pad={"r": 0, "t": 10},
                showactive=True,
                x=0.5,
                xanchor="left",
                y=1.2,
                yanchor="top",
            )
        ],
        width=700,
        height=500,
        title="Methods of differencing",
        title_x=0.12,
        title_y=0.93,
        margin=dict(t=60),
        legend=dict(yanchor="top", y=-0.1, xanchor="left", x=0.01),
    )
    return fig


def plot_acf_pacf(type="ar", nsample=200, nlags=20):
    np.random.seed(2021)
    if type == "ar":
        title = "AR(1) series"
        ar = 1
        ma = 0
    elif type == "ma":
        title = "MA(1) series"
        ar = 0
        ma = 1
    else:
        raise ("Please specify type as 'ar' or 'ma'.")
    raw_data = pd.DataFrame().assign(
        **{
            f"{coef:.2f}": arma_generate_sample(
                ar=[1, -coef * ar], ma=[1, coef * ma], nsample=nsample
            )
            for coef in np.arange(-0.8, 0.81, 0.2)
        }
    )
    acf_data = pd.DataFrame().assign(
        **{
            f"{col}": acf(raw_data[col], nlags=nlags, fft=True)
            for col in raw_data.columns
        }
    )
    pacf_data = pd.DataFrame().assign(
        **{
            f"{col}": pacf(raw_data[col], nlags=nlags)
            for col in raw_data.columns
        }
    )

    fig = make_subplots(rows=1, cols=3, subplot_titles=(title, "ACF","PACF"))
    fig.add_trace(
        go.Scatter(
            x=raw_data.index, y=raw_data.iloc[:, 0], mode="lines", name="Data"
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=acf_data.index,
            y=acf_data.iloc[:, 0],
            # width=[0.8] * len(acf_data),
            name="ACF",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=pacf_data.index,
            y=pacf_data.iloc[:, 0],
            # width=[0.8] * len(acf_data),
            name="PACF",
        ),
        row=1,
        col=3,
    )
    fig.update_xaxes(
        range=[0, nsample],
        tick0=0,
        row=1,
        col=1,
        title="Time",
        title_standoff=0,
    )
    fig.update_xaxes(
        range=[0, nlags],
        tick0=0,
        row=1,
        col=2,
        title="Lag",
        title_standoff=0,
    )
    fig.update_yaxes(
        range=[-10, 10],
        tick0=-10,
        dtick=5,
        row=1,
        col=1,
        title_standoff=0,
    )
    fig.update_yaxes(
        range=[-1.1, 1.1],
        tick0=-1,
        dtick=0.2,
        row=1,
        col=2,
        title_standoff=0,
    )
    frames = [
        dict(
            name=f"{col}",
            data=[
                go.Scatter(x=raw_data.index, y=raw_data[col]),
                go.Bar(x=acf_data.index, y=acf_data[col]),
                go.Bar(x=pacf_data.index, y=pacf_data[col]),
            ],
            traces=[0, 1, 2],
        )
        for col in raw_data.columns
    ]
    sliders = [
        {
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "phi=",
                "visible": True,
            },
            "pad": {"b": 10, "t": 30},
            "steps": [
                {
                    "args": [
                        [f"{col}"],
                        {
                            "frame": {
                                "duration": 0,
                                "easing": "linear",
                                "redraw": False,
                            },
                            "transition": {"duration": 0, "easing": "linear"},
                        },
                    ],
                    "label": f"{col}",
                    "method": "animate",
                }
                for col in raw_data.columns
            ],
        }
    ]
    fig.update(frames=frames)
    fig.update_layout(
        sliders=sliders,
        width=1000,
        height=450,
        showlegend=True,
        margin=dict(t=60),
    )
    return fig