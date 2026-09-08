"""
High-level execution utilities for offshore wind O&M analysis.

This module connects the component, simulation, economic, and visualization
parts of the :mod:`offshore_om` package. It provides functions for loading
scenario data, running Monte Carlo O&M simulations, evaluating economic
outcomes, constructing cumulative cost trajectories, visualizing cost risk,
and exporting consolidated results.

The functions in this module are intended to provide the main interface
between the underlying package and simulation notebooks.
"""

from pathlib import Path

import pandas as pd

from offshore_om.components import WindRegions, build_component_types
from offshore_om.constants import HOURS_PER_YEAR
from offshore_om.plotting import (
    build_cost_trajectories,
    plot_cost_risk_evolution,
)

from offshore_om.economics import (
    calculate_cfd_subsidy_capex,
    get_market_region_name,
    read_market_price_file,
    evaluate_monte_carlo_economics,
)
from offshore_om.simulation import simulate_wind_farm_OandM

import matplotlib.pyplot as plt


def load_scenario_configs(scenario_config):
    """
    Load market-price data for a collection of scenario configurations.

    Each input configuration is transformed into a dictionary containing the
    scenario name and the market data read from its associated market-price
    file.

    Parameters
    ----------
    scenario_config : iterable of dict
        Scenario configurations. Each dictionary must contain the keys
        ``"Scenario"`` and ``"MarketFile"``.

    Returns
    -------
    list of dict
        Loaded scenario configurations. Each returned dictionary contains:

        ``"Scenario"``
            Name of the market scenario.

        ``"MarketData"``
            Market-price data returned by
            :func:`offshore_om.economics.read_market_price_file`.
    """

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
    make_plots=False,
):
    """
    Simulate O&M and economic outcomes for one offshore wind region.

    The function runs the offshore wind O&M Monte Carlo model for the selected
    region and wind-farm configuration. It subsequently evaluates the economic
    consequences under the supplied market, discount-rate, and subsidy-price
    scenarios.

    Cumulative direct-cost trajectories and cost-risk figures are also
    generated for the region. The resulting economic rows are supplemented
    with technical, geographic, and simulation metadata.

    Parameters
    ----------
    region : WindRegion
        Offshore wind region to simulate. The object must provide the
        attributes ``name``, ``floating``, ``capacity_factor``, and
        ``distance_to_shore_km``.
    scenario_configs : iterable of dict
        Loaded market scenario configurations. Each dictionary must contain
        the keys ``"Scenario"`` and ``"MarketData"``.
    component_types : list
        Component-type definitions assigned to each wind turbine.
    n_turbines : int
        Number of turbines in the simulated wind farm.
    capacity_per_turbine : float
        Rated capacity of one turbine in MW.
    horizon_years : float
        Simulation horizon in years.
    failure_rate_type : str
        Name or identifier of the failure-rate assumption used to construct
        the component types.
    discount_rate_list : iterable of float
        Discount rates included in the economic evaluation.
    subsidy_price_list : iterable of float, optional
        Subsidy strike prices in ore per kWh. If omitted, the economic
        evaluation is performed without an explicitly supplied list.

    Returns
    -------
    list of dict
        Economic evaluation rows containing cost metrics and corresponding
        technical and scenario metadata.

    Notes
    -----
    Direct, lost-production, and total costs are normalized by installed
    wind-farm capacity before being returned.

    Availability is calculated from the downtime statistic associated with
    each economic metric and is bounded below by zero.

    This function also exports cumulative cost-risk and terminal cost
    distribution figures for the selected region.
    """

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

    if make_plots:
        t_grid, trajectories = build_cost_trajectories(
            events_by_simulation=sim_result["events_by_simulation"],
            horizon_years=sim_result["horizon_years"],
            n_points=1000,
        )

        fig, ax, fig_hist, ax_hist = plot_cost_risk_evolution(
            t_grid=t_grid,
            trajectories=trajectories,
            output_name=region.name,
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
    """
    Calculate and export subsidy-related CAPEX adjustments for TIMES scenarios.

    The function evaluates every combination of TIMES scenario, discount rate,
    and subsidy price. The individual results are combined into one table and
    exported to an Excel workbook.

    Parameters
    ----------
    times_subsidy_config : dict
        Configuration used for the subsidy calculation. It must contain
        ``"region_price_map"`` and may contain ``"file_pattern"`` and
        ``"process_filter"``.
    discount_rate_list : iterable of float
        Discount rates used in the subsidy CAPEX calculations.
    subsidy_price_list : iterable of float
        Subsidy prices in ore per kWh.
    times_scenario : iterable of dict
        TIMES scenario configurations. Each dictionary must contain
        ``"Scenario"``, ``"production_folder"``, ``"market_price_file"``,
        and ``"capacity_file"``.
    base_year : int
        Base year used for discounting.
    output_dir : path-like
        Directory in which the resulting Excel workbook is written.

    Returns
    -------
    pandas.DataFrame
        Combined subsidy CAPEX results for all TIMES scenarios, discount
        rates, and subsidy prices.

    Notes
    -----
    The returned table contains the additional columns ``"Scenario"``,
    ``"SubsidyPrice"``, and ``"DiscountRate"``.

    The combined results are exported as ``subsidy_capex.xlsx`` in
    ``output_dir``.
    """
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
    make_plots=False,
):
    """
    Run the complete offshore wind O&M scenario matrix.

    The function evaluates combinations of turbine capacity, failure-rate
    assumption, offshore wind region, wind-farm size, and simulation horizon.
    For each combination, it runs the regional O&M and economic analysis and
    collects the resulting rows in a consolidated table.

    A direct-cost comparison figure is produced for the selected site, and the
    complete result table is exported to Excel.

    Parameters
    ----------
    scenario_configs : iterable of dict
        Loaded market scenario configurations. Each dictionary must contain
        the keys ``"Scenario"`` and ``"MarketData"``.
    turbine_capacity_list : iterable of float
        Turbine capacities in MW to include in the scenario matrix.
    n_turbines_list : iterable of int
        Wind-farm sizes, expressed as numbers of turbines.
    discount_rate_list : iterable of float
        Discount rates included in the economic evaluation.
    horizon_years : iterable of float
        Simulation horizons in years.
    failure_rate_types : iterable of str
        Failure-rate assumptions used to construct component types.
    subsidy_price_list : iterable of float
        Subsidy prices in ore per kWh included in the economic evaluation.
    output_dir : pathlib.Path, optional
        Output location used when constructing the master-table Excel path.
        The default is ``Path("om_results_master_table.xlsx")``.

    Returns
    -------
    pandas.DataFrame
        Consolidated technical and economic results for the complete scenario
        matrix.

    Notes
    -----
    Component types are rebuilt for each combination of turbine capacity,
    failure-rate assumption, and fixed or floating wind-region configuration.

    The exported master table places the main scenario descriptors, technical
    parameters, and cost metrics before any remaining columns.

    A direct O&M cost comparison figure is exported as
    ``figures/direct_cost_vs_farm_size_metrics.pdf``.
    """

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
                            make_plots=make_plots,
                        )

                        all_rows.extend(rows)
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



