
"""
Visualization utilities for offshore wind O&M risk results.

This module contains plotting functions used to analyse and
communicate the translation of O&M uncertainty into cost and
availability metrics for offshore wind farms. It provides functions
to visualize the evolution of cumulative direct costs over time, the 
composition of annualized O&M costs, and the translation of 
availability factors into downtime estimates.  
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def build_cost_trajectories(
    events_by_simulation,
    horizon_years,
    n_points=1000,
):
    """
    Construct cumulative direct-cost trajectories from simulation events.

    Each Monte Carlo realization is converted into a stepwise cumulative-cost
    trajectory on a common time grid. The timing of individual cost events is
    preserved, and each trajectory remains constant between consecutive
    events.

    Parameters
    ----------
    events_by_simulation : list of list
        Cost events grouped by Monte Carlo realization. Each event must expose
        the attributes ``time_years`` and ``direct_cost``.
    horizon_years : float
        Simulation horizon in years.
    n_points : int, optional
        Number of points in the common time grid. The default is 1000.

    Returns
    -------
    t_grid : numpy.ndarray
        Common time grid extending from zero to ``horizon_years``. The array
        has shape ``(n_points,)``.
    trajectories : numpy.ndarray
        Cumulative undiscounted direct-cost trajectories. The array has shape
        ``(n_simulations, n_points)``.

    Notes
    -----
    Events outside the interval from zero to ``horizon_years`` are excluded.
    Simulations without valid events retain a zero-valued trajectory.
    """

    t_grid = np.linspace(
        0.0,
        float(horizon_years),
        n_points,
    )

    n_simulations = len(events_by_simulation)

    trajectories = np.zeros(
        (n_simulations, n_points),
        dtype=float,
    )

    for sim_id, events in enumerate(events_by_simulation):

        # Retain only events inside the simulation horizon
        valid_events = [
            event
            for event in events
            if 0.0 <= event.time_years <= horizon_years
        ]

        # Sort events chronologically
        valid_events = sorted(
            valid_events,
            key=lambda event: event.time_years,
        )

        if not valid_events:
            continue

        event_times = np.asarray(
            [event.time_years for event in valid_events],
            dtype=float,
        )

        event_costs = np.asarray(
            [event.direct_cost for event in valid_events],
            dtype=float,
        )

        cumulative_event_costs = np.cumsum(event_costs)

        # For every grid point, locate the latest event that has occurred.
        # This produces a true stepwise cumulative-cost trajectory.
        event_indices = (
            np.searchsorted(
                event_times,
                t_grid,
                side="right",
            )
            - 1
        )

        event_has_occurred = event_indices >= 0

        trajectories[
            sim_id,
            event_has_occurred,
        ] = cumulative_event_costs[
            event_indices[event_has_occurred]
        ]

    return t_grid, trajectories


def plot_cost_risk_evolution(
    t_grid,
    trajectories,
    output_name = "region",
    currency_scale=1e6,
    currency_label="MNOK",
    show_fan_bands=True,
):
    """
    Plot cumulative direct-cost risk over time and at the final horizon.

    The first figure presents the time-dependent mean, 75th percentile,
    95 percent value at risk, and 95 percent conditional value at risk of
    cumulative undiscounted direct costs. Optional uncertainty bands show the
    5th to 95th and 25th to 75th percentile ranges.

    The second figure presents the distribution of cumulative direct costs at
    the final point of the simulation horizon.

    Parameters
    ----------
    t_grid : array-like
        One-dimensional time grid in years.
    trajectories : array-like
        Two-dimensional array of cumulative direct-cost trajectories. Rows
        represent Monte Carlo realizations and columns represent time points.
    output_name : str, optional
        Name appended to the exported figure filenames. The default is
        ``"region"``.
    currency_scale : float, optional
        Divisor used to scale the cost values for visualization. For example,
        ``1e6`` converts currency units to millions. The default is ``1e6``.
    currency_label : str, optional
        Currency unit displayed on figure axes and legend entries. The default
        is ``"MNOK"``.
    show_fan_bands : bool, optional
        Whether to display the percentile uncertainty bands. The default is
        ``True``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the cumulative cost-risk evolution.
    ax : matplotlib.axes.Axes
        Axes containing the cumulative cost-risk evolution.
    fig_hist : matplotlib.figure.Figure
        Figure containing the final cost distribution.
    ax_hist : matplotlib.axes.Axes
        Axes containing the final cost distribution.

    Raises
    ------
    ValueError
        If ``trajectories`` is not two-dimensional.
    ValueError
        If the number of trajectory columns differs from the length of
        ``t_grid``.
    ValueError
        If ``currency_scale`` is less than or equal to zero.

    Notes
    -----
    At each time point, VaR95 is calculated as the 95th percentile of
    cumulative cost. CVaR95 is calculated as the mean of costs greater than
    or equal to the contemporaneous VaR95 threshold.

    The figures are exported as PDF files to the ``figures`` directory.
    """
    paths = {
        "evolution": Path(
            f"figures/cumulative_direct_cost_risk_evolution_{output_name}.pdf"
        ),
        "histogram": Path(
            f"figures/hist_direct_cost_distribution_{output_name}.pdf"
        ),
    }
    
    t_grid = np.asarray(t_grid, dtype=float)
    trajectories = np.asarray(trajectories, dtype=float)

    if trajectories.ndim != 2:
        raise ValueError(
            "trajectories must be a two-dimensional array with shape "
            "(n_simulations, n_time_points)."
        )

    if trajectories.shape[1] != len(t_grid):
        raise ValueError(
            "The number of trajectory columns must equal the length "
            "of t_grid."
        )

    if currency_scale <= 0:
        raise ValueError("currency_scale must be greater than zero.")

    costs = trajectories / currency_scale

    # Time-dependent Monte Carlo statistics
    mean = np.mean(costs, axis=0)
    p05 = np.quantile(costs, 0.05, axis=0)
    p25 = np.quantile(costs, 0.25, axis=0)
    p75 = np.quantile(costs, 0.75, axis=0)
    var95 = np.quantile(costs, 0.95, axis=0)

    # CVaR95 is the mean cost in the upper 5% tail at each time point
    cvar95 = np.empty_like(var95)

    for time_id, threshold in enumerate(var95):

        tail_costs = costs[:, time_id][
            costs[:, time_id] >= threshold
        ]

        cvar95[time_id] = (
            np.mean(tail_costs)
            if tail_costs.size > 0
            else threshold
        )

    final_mean = mean[-1]
    final_std = np.std(costs[:, -1], ddof=1)
    risk_ratio = cvar95[-1] / mean[-1]


    fig, ax = plt.subplots(figsize=(9, 5.5))

    if show_fan_bands:

        ax.fill_between(
            t_grid,
            p05,
            var95,
            step="post",
            color="#B8D8E8",
            alpha=0.45,
            linewidth=0,
            label="p5–p95 range",
        )

        ax.fill_between(
            t_grid,
            p25,
            p75,
            step="post",
            color="#4F9EC4",
            alpha=0.45,
            linewidth=0,
            label="p25–p75 range",
        )

    ax.step(
        t_grid,
        mean,
        where="post",
        color="#003B5C",
        linewidth=2.3,
        label=f"Mean ({mean[-1]:.1f} {currency_label})",
    )

    ax.step(
        t_grid,
        p75,
        where="post",
        color="#3786A6",
        linewidth=1.9,
        label=f"p75 ({p75[-1]:.1f} {currency_label})",
    )

    ax.step(
        t_grid,
        var95,
        where="post",
        color="#D95F02",
        linewidth=2.1,
        label=fr"VaR$_{95} ({var95[-1]:.1f} {currency_label})$",
    )

    ax.step(
        t_grid,
        cvar95,
        where="post",
        color="#8B1A1A",
        linewidth=2.3,
        linestyle="--",
        label=fr"CVaR$_{95} ({cvar95[-1]:.1f} {currency_label})$",
    )

    ax.plot([], [], ' ', label=f"Std = {final_std:.1f} {currency_label}")
    ax.plot([], [], ' ', label=f"CV = {final_std/final_mean:.1%}")

    ax.set_xlabel("Year")

    ax.set_ylabel(
        f"Cumulative undiscounted direct cost [{currency_label}]"
    )

    ax.set_xlim(
        0.0,
        float(np.max(t_grid)),
    )

    ax.set_ylim(bottom=0.0)

    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.7,
        alpha=0.6,
    )

    ax.legend(
        frameon=False,
        loc="upper left",
        ncol=2,
    )

    fig.tight_layout()


    fig.savefig(
        paths["evolution"],
        bbox_inches="tight",
        dpi=300,
    )

    print(f"Figure exported to: {paths["evolution"].resolve()}")

    final_costs = costs[:, -1]

    final_mean = np.mean(final_costs)
    final_std = np.std(final_costs, ddof=1)
    final_p75 = np.percentile(final_costs, 75)
    final_var95 = np.percentile(final_costs, 95)
    final_cvar95 = final_costs[
        final_costs >= final_var95
    ].mean()

    fig_hist, ax_hist = plt.subplots(figsize=(7, 4))

    ax_hist.hist(
        final_costs,
        bins=60,
        edgecolor="black",
        alpha=0.7,
    )

    ax_hist.axvspan(
        final_var95,
        final_costs.max(),
        color="#D95F02",
        alpha=0.15,
        label="Worst 5%",
    )

    ax_hist.axvline(
        final_mean,
        color="#003B5C",
        linewidth=2,
        label=f"Mean = {final_mean:.1f}",
    )

    ax_hist.axvline(
        final_p75,
        color="#3786A6",
        linestyle="--",
        linewidth=2,
        label=f"P75 = {final_p75:.1f}",
    )

    ax_hist.axvline(
        final_var95,
        color="#D95F02",
        linestyle="--",
        linewidth=2,
        label=f"VaR95 = {final_var95:.1f}",
    )

    ax_hist.axvline(
        final_cvar95,
        color="#8B1A1A",
        linestyle="--",
        linewidth=2,
        label=f"CVaR95 = {final_cvar95:.1f}",
    )

    ax_hist.set_xlabel(
        f"Lifetime direct O&M cost [{currency_label}]"
    )

    ax_hist.set_ylabel("Number of simulations")

    ax_hist.legend(frameon=False)

    fig_hist.tight_layout()

    fig_hist.savefig(
        paths["histogram"],
        bbox_inches="tight",
        dpi=300,
    )

    print(f"Figure exported to: {paths["histogram"].resolve()}")

    return fig, ax, fig_hist, ax_hist

def plot_direct_cost_vs_farm_size(
    master_df,
    output_name="direct_cost_vs_farm_size_metrics",
    site="SorvestF",
):
    """
    Plot direct O&M cost as a function of wind farm size.

    Creates a line plot showing how annualized direct O&M costs vary
    with the number of turbines for different risk metrics. Each risk
    metric is displayed as a separate line to facilitate comparison of
    scaling effects across risk levels.

    Parameters
    ----------
    master_df : pandas.DataFrame
        Summary results dataframe containing direct O&M cost metrics.
        Must contain at least the columns 'Site', 'Metric',
        'ParkSize', and 'DirectCost'.
    output_name : str, default="direct_cost_vs_farm_size_metrics"
        Suffix used when saving the figure.
    site : str, default="SorvestF"
        Offshore wind site to visualize.

    Returns
    -------
    tuple
        Matplotlib ``(fig, ax)`` containing the generated figure and
        axis.

    Notes
    -----
    Direct O&M costs are plotted in units of
    kNOK/MW/year and grouped by risk metric.
    """
    plot_df = master_df.loc[
        master_df["Site"].eq(site)
    ].copy()

    fig, ax = plt.subplots(figsize=(7, 4))

    for metric, grp in plot_df.groupby("Metric"):
        grp = grp.sort_values("ParkSize")

        ax.plot(
            grp["ParkSize"],
            grp["DirectCost"],
            marker="o",
            linewidth=2,
            label=metric,
        )

    ax.set_xlabel("Number of turbines")
    ax.set_ylabel("Direct O&M cost [kNOK/MW/year]")

    ax.legend(title="Metric")

    fig.tight_layout()

    output_path = Path(
        f"figures/{output_name}.pdf"
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        bbox_inches="tight",
        dpi=300,
    )

    print(
        f"Figure exported to: {output_path.resolve()}"
    )

    return fig, ax

def plot_cost_composition(
    master_df,
    output_name,
    scenario="TECH market",
    sites=("Nordavind", "SorvestF"),
):
    """
    Plot the composition of annualized O&M costs across risk metrics.

    Creates stacked bar charts showing the contribution of direct
    maintenance costs and lost-production costs for selected offshore
    wind sites. Results are displayed for the Mean, P75, P95, and
    CVaR95 risk metrics and exported as a PDF figure.

    Parameters
    ----------
    master_df : pandas.DataFrame
        Summary results dataframe containing O&M cost statistics.
        Must contain at least the columns 'Site', 'Metric',
        'DirectCost', 'LostCost', and the scenario/filter columns
        used internally by the function.
    output_name : str
        Suffix used when saving the figure.
    scenario : str, default="TECH market"
        Scenario to visualize.
    sites : tuple[str], default=("Nordavind", "SorvestF")
        Sites included in the comparison.

    Returns
    -------
    tuple
        Matplotlib ``(fig, axes)`` containing the generated figure and
        subplot axes.

    Notes
    -----
    The function applies a fixed reference configuration:
    100 MW wind farm, 15 MW turbines, 6% discount rate, and the
    Hendriks failure-rate model.
    """

    metric_order = ["Mean", "P75", "P95", "CVaR95"]

    plot_df = master_df.loc[
        master_df["Site"].isin(sites)
        & master_df["Metric"].isin(metric_order)
        & master_df["Scenario"].eq(scenario)
        & master_df["ParkSize"].eq(100)
        & master_df["TurbineCapacity"].eq(15)
        & master_df["DiscountRate"].eq(0.06)
        & master_df["FailureRateType"].eq("hendriks")
    ].copy()

    plot_df["Metric"] = pd.Categorical(
        plot_df["Metric"],
        categories=metric_order,
        ordered=True,
    )

    plot_df = plot_df.sort_values(["Site", "Metric"])

    fig, axes = plt.subplots(
        1,
        len(sites),
        figsize=(11, 4.8),
        sharey=True,
    )

    axes = np.atleast_1d(axes)

    for ax, site in zip(axes, sites):
        print("\n----------------")
        print(site)

        tmp = plot_df.loc[plot_df["Site"].eq(site)]

        print(tmp[[
            "Site",
            "Metric",
            "DiscountRate",
            "ParkSize",
            "TurbineCapacity",
            "FailureRateType",
            "Scenario"
        ]])

        print("\nDuplicates:")
        print(tmp.groupby("Metric").size())

        site_df = (
            plot_df.loc[plot_df["Site"].eq(site)]
            .set_index("Metric")
            .reindex(metric_order)
        )

        if site_df[["DirectCost", "LostCost"]].isna().any().any():
            raise ValueError(
                f"Missing cost data for {site} after filtering by "
                f"{scenario!r}."
            )

        x = np.arange(len(metric_order))

        ax.bar(
            x,
            site_df["DirectCost"],
            color="#3786A6",
            label="Direct cost",
        )

        ax.bar(
            x,
            site_df["LostCost"],
            bottom=site_df["DirectCost"],
            color="#D95F02",
            label="Lost-production cost",
        )

        # -------------------------------------
        # Add percentage labels inside bars
        # -------------------------------------

        direct = site_df["DirectCost"].values
        lost = site_df["LostCost"].values
        total = direct + lost

        for xpos, d, l, t in zip(x, direct, lost, total):

            direct_share = d / t
            lost_share = l / t

            # Direct-cost percentage
            ax.text(
                xpos,
                d / 2,
                f"{direct_share:.1%}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
            )

            # Lost-production percentage
            ax.text(
                xpos,
                d + l / 2,
                f"{lost_share:.1%}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
            )

        ax.set_title(site)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_order)
        ax.set_xlabel("Risk metric")
        ax.grid(
            axis="y",
            linestyle=":",
            linewidth=0.7,
            alpha=0.6,
        )

    axes[0].set_ylabel(
        "Annualized O&M cost [kNOK/MW/year]"
    )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.93))

    output_path = Path(
        f"figures/cost_composition_{output_name}.pdf"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        bbox_inches="tight",
        dpi=300,
    )

    print(f"Figure exported to: {output_path.resolve()}")

    return fig, axes


def plot_availability_loss(
    master_df,
    output_name,
    scenario="TECH market",
    sites=("Nordavind", "SorvestF"),
):
    """
    Plot availability changes relative to the mean-risk case.

    Creates bar charts showing the percentage change in availability
    factor for P75, P95, and CVaR95 risk metrics relative to the
    corresponding Mean value for each selected site.

    Parameters
    ----------
    master_df : pandas.DataFrame
        Summary results dataframe containing availability metrics.
        Must contain at least the columns 'Site', 'Metric',
        and 'AvailabilityFactor'.
    output_name : str
        Suffix used when saving the figure.
    scenario : str, default="TECH market"
        Scenario to visualize.
    sites : tuple[str], default=("Nordavind", "SorvestF")
        Sites included in the comparison.

    Returns
    -------
    tuple
        Matplotlib ``(fig, axes)`` containing the generated figure and
        subplot axes.

    Notes
    -----
    Availability impacts are reported as percentage deviations from
    the mean-risk case. Positive values indicate improved
    availability, while negative values indicate increased downtime.
    """

    metric_order = ["Mean", "P75", "P95", "CVaR95"]

    plot_df = master_df.loc[
        master_df["Site"].isin(sites)
        & master_df["Metric"].isin(metric_order)
        & master_df["Scenario"].eq(scenario)
        & master_df["ParkSize"].eq(100)
        & master_df["TurbineCapacity"].eq(15)
        & master_df["DiscountRate"].eq(0.06)
        & master_df["FailureRateType"].eq("hendriks")
    ].copy()

    plot_df["Metric"] = pd.Categorical(
        plot_df["Metric"],
        categories=metric_order,
        ordered=True,
    )

    fig, axes = plt.subplots(
        1,
        len(sites),
        figsize=(10, 4.5),
        sharey=True,
    )

    axes = np.atleast_1d(axes)

    for ax, site in zip(axes, sites):

        site_df = (
            plot_df.loc[plot_df["Site"] == site]
            .set_index("Metric")
            .reindex(metric_order)
        )

        mean_afa = site_df.loc[
            "Mean",
            "AvailabilityFactor",
        ]

        loss_pct = (
            (site_df["AvailabilityFactor"] - mean_afa)
            / mean_afa
            * 100
        )

        x = np.arange(len(metric_order))

        bars = ax.bar(
            x,
            loss_pct,
            color="#4F9EC4",
        )

        for bar, value in zip(bars, loss_pct):

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.2f}%",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )

        ax.axhline(
            0,
            color="black",
            linewidth=1,
        )

        ax.set_title(site)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_order)
        ax.set_xlabel("Risk metric")

        ax.grid(
            axis="y",
            linestyle=":",
            linewidth=0.7,
            alpha=0.6,
        )

    axes[0].set_ylabel(
        "Availability change relative to mean [%]"
    )

    fig.tight_layout()

    output_path = Path(
        f"figures/availability_loss_{output_name}.pdf"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        bbox_inches="tight",
        dpi=300,
    )

    print(
        f"Figure exported to: {output_path.resolve()}"
    )

    return fig, axes


def plot_availability_translation(
    master_df,
    output_name,
    scenario="TECH market",
    sites=("Nordavind", "SorvestF"),
):
    """
    Translate availability factors into annual downtime estimates.

    Creates bar charts showing the equivalent downtime in days per
    year associated with each risk metric. Downtime is calculated
    directly from the availability factor as
    ``(1 - AvailabilityFactor) * 365``.

    Parameters
    ----------
    master_df : pandas.DataFrame
        Summary results dataframe containing availability metrics.
        Must contain at least the columns 'Site', 'Metric',
        and 'AvailabilityFactor'.
    output_name : str
        Suffix used when saving the figure.
    scenario : str, default="TECH market"
        Scenario to visualize.
    sites : tuple[str], default=("Nordavind", "SorvestF")
        Sites included in the comparison.

    Returns
    -------
    tuple
        Matplotlib ``(fig, axes)`` containing the generated figure and
        subplot axes.

    Notes
    -----
    This visualization provides a more intuitive interpretation of
    availability impacts by expressing them as expected downtime
    days per year rather than availability factors.
    """

    metric_order = ["Mean", "P75", "P95", "CVaR95"]

    plot_df = master_df.loc[
        master_df["Site"].isin(sites)
        & master_df["Metric"].isin(metric_order)
        & master_df["Scenario"].eq(scenario)
        & master_df["ParkSize"].eq(100)
        & master_df["TurbineCapacity"].eq(15)
        & master_df["DiscountRate"].eq(0.06)
        & master_df["FailureRateType"].eq("hendriks")
    ].copy()

    plot_df["Metric"] = pd.Categorical(
        plot_df["Metric"],
        categories=metric_order,
        ordered=True,
    )

    fig, axes = plt.subplots(
        1,
        len(sites),
        figsize=(10, 4.5),
        sharey=True,
    )

    axes = np.atleast_1d(axes)

    for ax, site in zip(axes, sites):

        site_df = (
            plot_df.loc[plot_df["Site"] == site]
            .set_index("Metric")
            .reindex(metric_order)
        )

        downtime_days = (
            1 - site_df["AvailabilityFactor"]
        ) * 365

        x = np.arange(len(metric_order))

        bars = ax.bar(
            x,
            downtime_days,
            color="#4F9EC4",
        )

        for bar, value in zip(bars, downtime_days):

            ax.text(
                bar.get_x() + bar.get_width()/2,
                value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_title(site)

        ax.set_xticks(x)
        ax.set_xticklabels(metric_order)

        ax.set_xlabel("Risk metric")

        ax.grid(
            axis="y",
            linestyle=":",
            linewidth=0.7,
            alpha=0.6,
        )

    axes[0].set_ylabel(
        "Downtime [days/year]"
    )


    fig.tight_layout()

    output_path = Path(
        f"figures/availability_translation_{output_name}.pdf"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        bbox_inches="tight",
        dpi=300,
    )

    print(
        f"Figure exported to: {output_path.resolve()}"
    )

    return fig, axes