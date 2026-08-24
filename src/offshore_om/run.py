from pathlib import Path

import pandas as pd

from offshore_om.components import WindRegions, build_component_types
from offshore_om.constants import HOURS_PER_YEAR

from offshore_om.economics import (
    calculate_cfd_subsidy_capex,
    convert_ore_kwh_to_knok_mwh,
    get_market_region_name,
    read_market_price_file,
    evaluate_monte_carlo_economics,
)
from offshore_om.simulation import simulate_wind_farm_OandM

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_cost_trajectories(
    events_by_simulation,
    horizon_years,
    n_points=1000,
):
    """
    Construct cumulative undiscounted direct-cost trajectories while
    preserving the timing of individual failure events.

    Parameters
    ----------
    events_by_simulation : list
        One list of cost events for each Monte Carlo simulation.
        Each event must contain `time_years` and `direct_cost`.

    horizon_years : float
        Simulation horizon in years.

    n_points : int, default=1000
        Number of points in the common time grid used to calculate
        the time-dependent Monte Carlo statistics.

    Returns
    -------
    t_grid : np.ndarray
        Common time grid from year 0 to the simulation horizon.

    trajectories : np.ndarray
        Cumulative undiscounted direct costs.
        Shape: (n_simulations, n_points).
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
    output_path="figures/cumulative_direct_cost_risk_evolution.pdf",
    currency_scale=1e6,
    currency_label="MNOK",
    show_fan_bands=True,
):
    """
    Plot the evolution of cumulative undiscounted direct-cost risk.

    The figure includes:
    - Mean
    - P75
    - VaR95, equivalent to P95 for direct costs
    - CVaR95, calculated as the mean above the contemporaneous VaR95
    - Optional P5-P95 and P25-P75 uncertainty bands
    """

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
            label="P5–P95 range",
        )

        ax.fill_between(
            t_grid,
            p25,
            p75,
            step="post",
            color="#4F9EC4",
            alpha=0.45,
            linewidth=0,
            label="P25–P75 range",
        )

    ax.step(
        t_grid,
        mean,
        where="post",
        color="#003B5C",
        linewidth=2.3,
        label="Mean",
    )

    ax.step(
        t_grid,
        p75,
        where="post",
        color="#3786A6",
        linewidth=1.9,
        label="P75",
    )

    ax.step(
        t_grid,
        var95,
        where="post",
        color="#D95F02",
        linewidth=2.1,
        label=r"VaR$_{95}$",
    )

    ax.step(
        t_grid,
        cvar95,
        where="post",
        color="#8B1A1A",
        linewidth=2.3,
        linestyle="--",
        label=r"CVaR$_{95}$",
    )

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

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        bbox_inches="tight",
        dpi=300,
    )

    print(f"Figure exported to: {output_path.resolve()}")

    return fig, ax

def load_scenario_configs(scenario_config):
    scenario_configs = []
    for config in scenario_config:
        scenario_configs.append(
            {
                "Scenario": config["Scenario"],
                "MarketData": read_market_price_file(config["MarketFile"]),
            }
        )
    return scenario_configs

def simulate_region_scenario(
    region,
    scenario_configs,
    component_types,
    n_turbines,
    capacity_per_turbine,
    horizon_years,
    failure_rate_type,
    discount_rate_list,
    subsidy_price_list=None,
):

    region_name = get_market_region_name(region.name)

    sim_result = simulate_wind_farm_OandM(
        n_turbines=n_turbines,
        component_types=component_types,
        region=region,
        capacity_per_turbine=capacity_per_turbine,
        horizon_years=horizon_years,
        daily_rate=10,
        n_simulations=1000,
        random_seed=123,
        part_capacity=3,
        resupply_time_h=24,
    )

    market_scenarios = {
        config["Scenario"]: config["MarketData"]
        for config in scenario_configs
    }

    econ_rows = evaluate_monte_carlo_economics(
        mc_result=sim_result,
        discount_rates=discount_rate_list,
        horizon_years=horizon_years,
        region_name=region_name,
        market_scenarios=market_scenarios,
        subsidy_prices_ore_per_kwh=subsidy_price_list,
    )
    
    t_grid, trajectories = build_cost_trajectories(
        events_by_simulation=sim_result["events_by_simulation"],
        horizon_years=sim_result["horizon_years"],
        n_points=1000,
    )

    fig, ax = plot_cost_risk_evolution(
        t_grid=t_grid,
        trajectories=trajectories,
        output_path=f"figures/cumulative_direct_cost_risk_evolution_{region.name}.pdf",
        currency_scale=1e6,
        currency_label="MNOK",
    )

    downtime_stats = sim_result["downtime"]
    installed_capacity = sim_result["installed_capacity"]

    for row in econ_rows:
        metric = row["Metric"]
        key = metric

        downtime = downtime_stats[key] / n_turbines

        availability_factor = max(
            0.0,
            1.0 - downtime / (HOURS_PER_YEAR * horizon_years)
        )

        # Convert to kNOK/MW/year
        row["DirectCost"] /= installed_capacity
        row["LostCost"] /= installed_capacity
        row["TotalCost"] /= installed_capacity

        row.update(
            {
                "Site": region.name,
                "ParkSize": n_turbines,
                "TurbineCapacity": capacity_per_turbine,
                "Horizon": horizon_years,
                "MarketRegion": region_name,
                "Floating": region.floating,
                "CapacityFactor": region.capacity_factor,
                "DistanceToShoreKm": region.distance_to_shore_km,
                "InstalledCapacity": installed_capacity,
                "FailureRateType": failure_rate_type,
                "AvailabilityFactor": availability_factor,
            }
        )

    return econ_rows

def run_subsidy_capex(times_subsidy_config, discount_rate_list, subsidy_price_list, times_scenario, base_year, output_dir):
    capex_dfs = []

    for scenario in times_scenario:
        for discount_rate in discount_rate_list:
            for subsidy_price_ore_per_kwh in subsidy_price_list:
                capex_df = calculate_cfd_subsidy_capex(
                    production_folder=Path(scenario["production_folder"]),
                    market_price_file=Path(scenario["market_price_file"]),
                    capacity_file=Path(scenario["capacity_file"]),
                    subsidy_price_ore_kwh=subsidy_price_ore_per_kwh,
                    region_price_map=times_subsidy_config["region_price_map"],
                    discount_rate=discount_rate,
                    base_year=base_year,
                    file_pattern=times_subsidy_config.get("file_pattern", "*.csv"),
                    process_filter=times_subsidy_config.get("process_filter"),
                )

                capex_df["Scenario"] = scenario["Scenario"]
                capex_df["SubsidyPrice"] = subsidy_price_ore_per_kwh
                capex_df["DiscountRate"] = discount_rate
                capex_dfs.append(capex_df)


    times_capex_df = pd.concat(capex_dfs, ignore_index=True)
        
    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    times_output_path = output_folder / "subsidy_capex.xlsx"
    
    times_capex_df.to_excel(times_output_path, index=False)

    return times_capex_df


def run_master(
    scenario_configs,
    turbine_capacity_list,
    n_turbines_list,
    discount_rate_list,
    horizon_years,
    failure_rate_types,
    subsidy_price_list,
    output_dir=Path("om_results_master_table.xlsx"),
):
    all_rows = []

    for capacity_per_turbine in turbine_capacity_list:
        for failure_rate_type in failure_rate_types:
            for region in WindRegions:
                print(f"Simulating for region: {region.name}")
                component_types = build_component_types(
                        capacity_per_turbine,
                        failure_rate_type,
                        region.floating
                    )

                for n_turbines in n_turbines_list:
                    for horizon in horizon_years:
                        rows = simulate_region_scenario(
                            region=region,
                            scenario_configs=scenario_configs,
                            component_types=component_types,
                            n_turbines=n_turbines,
                            capacity_per_turbine=capacity_per_turbine,
                            horizon_years=horizon,
                            failure_rate_type=failure_rate_type,
                            discount_rate_list=discount_rate_list,
                            subsidy_price_list=subsidy_price_list,
                        )

                        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)

    mean_df = df[df["Metric"] == "Mean"]

    fig, ax = plt.subplots(figsize=(7,4))

    for site, grp in mean_df.groupby("Site"):

        grp = grp.sort_values("ParkSize")

        ax.plot(
            grp["ParkSize"],
            grp["DirectCost"],
            marker="o",
            linewidth=2,
            label=site,
        )

    ax.set_xlabel("Number of turbines")
    ax.set_ylabel("Direct O&M cost [kNOK/MW/year]")

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        "figures/direct_cost_vs_farm_size_comparison.pdf",
        bbox_inches="tight",
    )

    master_df = pd.DataFrame(all_rows)

    preferred_columns = [
        "Site",
        "ParkSize",
        "Metric",
        "DiscountRate",
        "TurbineCapacity",
        "Horizon",
        "SubsidyPrice",
        "Scenario",
        "MarketRegion",
        "Floating",
        "CapacityFactor",
        "DistanceToShoreKm",
        "InstalledCapacity",
        "FailureRateType",
        "DirectCost",
        "LostCost",
        "TotalCost",
        "AvailabilityFactor",
    ]

    remaining_columns = [
        col
        for col in master_df.columns
        if col not in preferred_columns
    ]

    master_df = master_df[
        preferred_columns + remaining_columns
    ]

    output_path = Path(output_dir / "om_results_master_table.xlsx")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_excel(output_path, index=False)

    return master_df



