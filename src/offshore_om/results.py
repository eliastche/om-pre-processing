import pandas as pd


def results_to_master_table(
    results,
    regions,
    variable_list,
    scenario_label,
    discount_rate,
    turbine_capacity,
    filename="om_results_master.xlsx",
):
    rows = []
    for region in regions:
        site = region.name
        park_results = results[site]

        for variable in variable_list:
            sim_result = park_results[variable]
            eq_annual_stats = sim_result["equivalent_annual_costs"]
            installed_capacity = sim_result["installed_capacity"]

            metrics = {
                "mean": eq_annual_stats["mean"] / installed_capacity,
                "P75": eq_annual_stats["p75"] / installed_capacity,
                "P95": eq_annual_stats["p95"] / installed_capacity,
                "CVaR 95% (worst)": eq_annual_stats["cvar95"] / installed_capacity,
            }

            for metric_name, value in metrics.items():
                rows.append(
                    {
                        "Site": site,
                        "ParkSize": variable,
                        "Metric": metric_name,
                        "Value": value,
                        "DiscountRate": discount_rate,
                        "TurbineCapacity": turbine_capacity,
                        "Scenario": scenario_label,
                        "Floating": region.floating,
                        "CapacityFactor": region.capacity_factor,
                        "DistanceToShoreKm": region.distance_to_shore_km,
                    }
                )

    df = pd.DataFrame(rows)
    df = df[
        [
            "Site",
            "ParkSize",
            "Metric",
            "Value",
            "DiscountRate",
            "TurbineCapacity",
            "Scenario",
            "Floating",
            "CapacityFactor",
            "DistanceToShoreKm",
        ]
    ]
    df.to_excel(filename, index=False)
    return df
