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



