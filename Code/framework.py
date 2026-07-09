from dataclasses import dataclass
import pandas as pd
import numpy as np

kNOKEUR = 10.39/1000

#ComponentType(name="Blades", shape=0.75, scale=86.8 * 0.8,  repair_time=147.0, direct_cost=375689 * kNOKEUR),
#ComponentType(name="Gearbox", shape=1.38, scale=15.02 * 0.8, repair_time=261.0, direct_cost=647101 * kNOKEUR),
#ComponentType(name="Generator", shape=1.52, scale=18.72 * 0.8, repair_time=126.0, direct_cost=232634 * kNOKEUR),

@dataclass
class ComponentType:
    """
    Data structure for a wind turbine component's reliability and cost parameters.
    """
    name: str           # Name of the component (e.g., 'Gearbox')
    shape: float        # Weibull shape parameter (k)
    scale: float        # Weibull scale parameter (λ) in years (characteristic life)
    repair_time: float  # Average downtime for a failure (in hours)
    direct_cost: float  # Direct repair/replacement cost per failure (monetary units)
    # Additional fields (e.g., variable maintenance cost, cost distributions) can be added if needed.

@dataclass
class WindRegion:
    name: str 
    offshore_multiplyer: float
    capacity_factor: float
    cost_multiplyer: float


def sample_failure_times(component: ComponentType, horizon_years: float, rng: np.random.Generator, offshore_factor: float = 1.0):
    """
    Generate failure event times for a single component over a given horizon.
    Returns a list of failure times (years from start) for that component.
    Each failure is assumed to be followed by replacement, resetting the component's life (renewal process).
    """
    t = 0.0
    times = []
    while True:
        # Draw time to next failure from Weibull distribution (with shape and scale from component)
        # np.random.Generator.weibull(k) generates a Weibull-distributed sample with shape k and scale = 1,
        # so we multiply by the scale parameter to get the actual time.
        dt = rng.weibull(component.shape) * (component.scale/ (offshore_factor ** (1.0 / component.shape)))
        t += dt
        if t > horizon_years:
            break
        times.append(t)
    return times

def simulate_wind_farm_OandM(
    n_turbines: int,
    component_types: list = None,
    horizon_years: float = 25.0,
    number_of_parks: float = 1.0,
    capacity_per_turbine: float = 15.0,
    capacity_factor: float = 0.5,
    price_per_mwh: float = 125.11 *kNOKEUR,
    discount_rate: float = 0.07,
    n_simulations: int = 10000,
    random_seed: int = None,
    offshore_factor: float = 1.0,
    cost_multiplier: float = 1.0
):
    """
    Simulate the O&M cost for an offshore wind capacity using Monte Carlo, returning:
    (1) Lifetime total O&M cost (present value at time zero, as in original).
    (2) Cumulative O&M cost distributions at fixed 5-year target years, also discounted to present.
    (3) Marginal O&M cost distributions for each 5-year period, discounted to present.

    Returns a dictionary with original keys for lifetime cost (backward compatible), plus new keys:
      - 'target_years': list of target year offsets (e.g., [5, 10, 15, 20, 25]).
      - 'cumulative_costs_pv': dict mapping each target year offset to an array of cumulative O&M costs (present value) up to that year for all simulations.
      - 'cumulative_stats': dict mapping each target year offset to summary stats (mean, p50, var95, cvar95).
      - 'marginal_costs_pv': dict of arrays for costs within each period (e.g., '0-4', '5-9'), discounted to present.
      - 'marginal_stats': dict of summary stats for each 5-year period.
      (Original keys: 'total_costs', 'mean', 'p50', 'var95', 'cvar95', 'cost_per_mw' remain lifetime totals).
    """
    rng = np.random.default_rng(seed=random_seed)
    # Use default component types (key offshore turbine components) if none provided
    if component_types is None:
        component_types = [
            ComponentType(name="Blades", shape=0.75, scale=86.8,  repair_time=147.0, direct_cost=375689 * cost_multiplier * kNOKEUR),
            ComponentType(name="Gearbox", shape=1.38, scale=15.02, repair_time=261.0, direct_cost=647101 * cost_multiplier * kNOKEUR),
            ComponentType(name="Generator", shape=1.52, scale=18.72, repair_time=126.0, direct_cost=232634 * cost_multiplier * kNOKEUR),
            #ComponentType(name="ElectricalSystem", shape=1.0, scale=2.0 * 0.8, repair_time=24.0, direct_cost=5.0e4 * kNOKEUR)
        ]
    # Define target year offsets: every 5 years (plus final horizon if not exactly on a 5-year mark)
    max_year = int(np.ceil(horizon_years))
    target_offsets = list(range(5, max_year + 1, 5))
    if target_offsets and target_offsets[-1] < max_year:
        target_offsets.append(max_year)
    if not target_offsets:  # if horizon shorter than 5 years
        target_offsets = [max_year]
    # Initialize arrays for cumulative present-value costs up to each target year for all simulation runs
    cumulative_costs_pv_by_target = {offset: np.zeros(n_simulations) for offset in target_offsets}
    # Initialize array for total lifetime present-value costs (for each simulation run)
    total_costs_pv = np.zeros(n_simulations)
    # --- NEW: names for component-level reporting ---
    component_labels = [
        getattr(comp, "name", getattr(comp, "component_name", f"component_{i+1}"))
        for i, comp in enumerate(component_types)
    ]
    # --- NEW: lifetime totals per simulation ---
    total_direct_costs_pv = np.zeros(n_simulations)
    total_lost_prod_costs_pv = np.zeros(n_simulations)
    total_component_costs_pv = {
        label: np.zeros(n_simulations) for label in component_labels
    }
    # --- NEW: cumulative checkpoint totals per target offset ---
    cumulative_direct_costs_pv_by_target = {
        offset: np.zeros(n_simulations) for offset in target_offsets
    }
    cumulative_lost_prod_costs_pv_by_target = {
        offset: np.zeros(n_simulations) for offset in target_offsets
    }
    cumulative_component_costs_pv_by_target = {
        offset: {label: np.zeros(n_simulations) for label in component_labels}
        for offset in target_offsets
    }

    # Monte Carlo simulation
    for sim in range(n_simulations):
        # List of (event_time, event_cost_PV) for all failures in this simulation run
        events_pv = []
        for _ in range(n_turbines):
            for comp_idx, comp in enumerate(component_types):
                comp_label = component_labels[comp_idx]

                failure_times = sample_failure_times(comp, horizon_years, rng, offshore_factor)
                for t_fail in failure_times:
                    pv_factor = (1 + discount_rate) ** -t_fail

                    direct_cost = comp.direct_cost
                    lost_mwh = comp.repair_time * capacity_per_turbine * capacity_factor
                    lost_prod_cost = lost_mwh * price_per_mwh

                    direct_cost_pv = direct_cost * pv_factor
                    lost_prod_cost_pv = lost_prod_cost * pv_factor
                    event_cost_pv = direct_cost_pv + lost_prod_cost_pv

                    # store richer event tuple
                    events_pv.append(
                        (t_fail, comp_label, direct_cost_pv, lost_prod_cost_pv, event_cost_pv)
                    )
        # Sort events by failure time for consistent accumulation
        events_pv.sort(key=lambda e: e[0])
        # Accumulate present-value costs up to each target offset for this run
        cum_sum_pv = 0.0
        cum_direct_pv = 0.0
        cum_lost_prod_pv = 0.0
        cum_component_pv = {label: 0.0 for label in component_labels}

        offset_idx = 0

        for t_fail, comp_label, direct_pv, lost_prod_pv, total_pv in events_pv:
            # fill checkpoints before this event if event occurs after next target
            while offset_idx < len(target_offsets) and t_fail > target_offsets[offset_idx]:
                offset = target_offsets[offset_idx]

                cumulative_costs_pv_by_target[offset][sim] = cum_sum_pv
                cumulative_direct_costs_pv_by_target[offset][sim] = cum_direct_pv
                cumulative_lost_prod_costs_pv_by_target[offset][sim] = cum_lost_prod_pv

                for label in component_labels:
                    cumulative_component_costs_pv_by_target[offset][label][sim] = cum_component_pv[label]

                offset_idx += 1

            # accumulate current event
            cum_sum_pv += total_pv
            cum_direct_pv += direct_pv
            cum_lost_prod_pv += lost_prod_pv
            cum_component_pv[comp_label] += total_pv

        # fill any remaining checkpoints with final cumulative values
        while offset_idx < len(target_offsets):
            offset = target_offsets[offset_idx]

            cumulative_costs_pv_by_target[offset][sim] = cum_sum_pv
            cumulative_direct_costs_pv_by_target[offset][sim] = cum_direct_pv
            cumulative_lost_prod_costs_pv_by_target[offset][sim] = cum_lost_prod_pv

            for label in component_labels:
                cumulative_component_costs_pv_by_target[offset][label][sim] = cum_component_pv[label]

            offset_idx += 1

        # lifetime totals for this simulation
        total_costs_pv[sim] = cum_sum_pv
        total_direct_costs_pv[sim] = cum_direct_pv
        total_lost_prod_costs_pv[sim] = cum_lost_prod_pv

        for label in component_labels:
            total_component_costs_pv[label][sim] = cum_component_pv[label]

    # Scale up costs by the number of identical parks (assuming fully correlated operation across parks)
    if number_of_parks != 1.0:
        for offset in target_offsets:
            cumulative_costs_pv_by_target[offset] *= number_of_parks
            cumulative_direct_costs_pv_by_target[offset] *= number_of_parks
            cumulative_lost_prod_costs_pv_by_target[offset] *= number_of_parks

            for label in component_labels:
                cumulative_component_costs_pv_by_target[offset][label] *= number_of_parks

        total_costs_pv *= number_of_parks
        total_direct_costs_pv *= number_of_parks
        total_lost_prod_costs_pv *= number_of_parks

        for label in component_labels:
            total_component_costs_pv[label] *= number_of_parks
    # Compute marginal 5-year period costs by differencing cumulative costs between successive target offsets
    marginal_direct_costs_pv_by_period = {}
    marginal_lost_prod_costs_pv_by_period = {}
    marginal_component_costs_pv_by_period = {}

    prev_offset = 0
    for offset in target_offsets:
        period_start = prev_offset
        period_end = offset - 1 if offset > prev_offset else prev_offset
        period_label = f"{int(period_start)}-{int(period_end)}"

        direct_period_costs = cumulative_direct_costs_pv_by_target[offset].copy()
        lost_period_costs = cumulative_lost_prod_costs_pv_by_target[offset].copy()
        component_period_costs = {
            label: cumulative_component_costs_pv_by_target[offset][label].copy()
            for label in component_labels
        }

        if prev_offset > 0:
            direct_period_costs -= cumulative_direct_costs_pv_by_target[prev_offset]
            lost_period_costs -= cumulative_lost_prod_costs_pv_by_target[prev_offset]

            for label in component_labels:
                component_period_costs[label] -= cumulative_component_costs_pv_by_target[prev_offset][label]

        marginal_direct_costs_pv_by_period[period_label] = direct_period_costs
        marginal_lost_prod_costs_pv_by_period[period_label] = lost_period_costs
        marginal_component_costs_pv_by_period[period_label] = component_period_costs

        prev_offset = offset

    # Helper to compute summary statistics for an array of costs
    def summary_stats(cost_array):
        p_zero_cost = (cost_array == 0).mean()  # Fraction of runs with cost == 0

        # Central tendency
        mean_val = float(np.mean(cost_array))
        p50_val = float(np.quantile(cost_array, 0.50))

        # Percentiles (best → worst)
        p5_val  = float(np.quantile(cost_array, 0.05))   # best 5%
        p25_val = float(np.quantile(cost_array, 0.25))
        p75_val = float(np.quantile(cost_array, 0.75))
        p95_val = float(np.quantile(cost_array, 0.95))   # worst 5%

        # Tail means
        lower_tail = cost_array[cost_array <= p5_val]    # best-case tail
        upper_tail = cost_array[cost_array >= p95_val]   # worst-case tail

        cvar5_low  = float(np.mean(lower_tail)) if lower_tail.size > 0 else p5_val
        cvar95_high = float(np.mean(upper_tail)) if upper_tail.size > 0 else p95_val

        return {
            "p_zero_cost": p_zero_cost,
            "mean": mean_val,
            "p5": p5_val,
            "p25": p25_val,
            "p50": p50_val,
            "p75": p75_val,
            "p95": p95_val,
            "cvar5_low": cvar5_low,     # average of best 5% outcomes
            "cvar95_high": cvar95_high  # average of worst 5% outcomes
        }


    # Calculate statistics for the lifetime cost distribution (present value)
    # lifetime_stats = summary_stats(total_costs_pv)
    # Calculate statistics for each cumulative target-year distribution
    # cumulative_stats = {offset: summary_stats(cumulative_costs_pv_by_target[offset]) for offset in target_offsets}
    # Calculate statistics for each 5-year marginal period distribution
    marginal_stats = {period: summary_stats(marginal_direct_costs_pv_by_period[period]) for period in marginal_direct_costs_pv_by_period}
    marginal_lost_prod_stats = {period: summary_stats(marginal_lost_prod_costs_pv_by_period[period]) for period in marginal_lost_prod_costs_pv_by_period}   
    marginal_component_stats = {period: {label: summary_stats(marginal_component_costs_pv_by_period[period][label]) for label in component_labels} for period in marginal_component_costs_pv_by_period}
    
    marginal_total_costs_pv_by_period = {
        period: marginal_direct_costs_pv_by_period[period] + marginal_lost_prod_costs_pv_by_period[period]
        for period in marginal_direct_costs_pv_by_period
    }

    marginal_total_stats = {
        period: summary_stats(marginal_total_costs_pv_by_period[period])
        for period in marginal_total_costs_pv_by_period
    }

    # Prepare the results dictionary
    return {
        # Lifetime (full 0–horizon) present-value cost distribution
        "number_of_parks": number_of_parks,
        "n_turbines": n_turbines,
        "capacity_per_turbine": capacity_per_turbine,
        "total_production": number_of_parks * n_turbines * capacity_per_turbine,

        "lifetime_total_costs_pv": total_costs_pv,
        "lifetime_direct_costs_pv": total_direct_costs_pv,
        "lifetime_lost_prod_costs_pv": total_lost_prod_costs_pv,
        "lifetime_component_costs_pv": total_component_costs_pv,

        "lifetime_total_stats": summary_stats(total_costs_pv),
        "lifetime_direct_stats": summary_stats(total_direct_costs_pv),
        "lifetime_lost_prod_stats": summary_stats(total_lost_prod_costs_pv),
        "lifetime_component_stats": {
            label: summary_stats(total_component_costs_pv[label])
            for label in component_labels
            },

        # total marginal O&M
        "marginal_total_costs_pv": marginal_total_costs_pv_by_period,
        "marginal_total_stats": marginal_total_stats,

        # split views
        "marginal_direct_costs_pv": marginal_direct_costs_pv_by_period,
        "marginal_direct_stats": marginal_stats,
        "marginal_lost_prod_costs_pv": marginal_lost_prod_costs_pv_by_period,
        "marginal_lost_prod_stats": marginal_lost_prod_stats,
        "marginal_component_stats": marginal_component_stats,
    }


def pretty_print_results(results, currency="kNOK"):
    """
    Nicely print summary statistics from the O&M Monte Carlo simulation.

    Parameters
    ----------
    results : dict
        Output dictionary from simulate_wind_farm_OandM()
    currency : str
        Currency label for costs (e.g., 'NOK', 'EUR')
    """
    
    def fmt(x):
        return f"{x:,.0f}"

    print("\n" + "=" * 60)
    print(" Offshore Wind O&M Cost Risk Summary ")
    print("=" * 60)

    print(f"Number of turbines in park:    {fmt(results['n_turbines'])}")
    print(f"Capacity per turbine:          {fmt(results['capacity_per_turbine'])}")
    print(f"Total production:              {fmt(results['total_production'])} MW")
    print("=" * 60)
    print("Lifetime O&M Cost Breakdown (PV, discounted to today)")
    print("(per MW)")
    print("=" * 60)

    total_prod = results["total_production"]

    total_stats = results["lifetime_total_stats"]
    direct_stats = results["lifetime_direct_stats"]
    lost_stats = results["lifetime_lost_prod_stats"]

    # ---- TOTAL ----
    print("Expected TOTAL O&M")
    print("-" * 60)
    print(f"  Expected:              {fmt(total_stats['mean'])} {currency}")
    # ---- DIRECT vs LOST ----
    total_mean = total_stats["mean"]
    direct_mean = direct_stats["mean"]
    lost_mean = lost_stats["mean"]

    print("\nBREAKDOWN (mean shares)")
    print("-" * 60)

    if total_mean > 0:
        print(f"  Direct cost:       {100 * direct_mean / total_mean:.1f}%")
        print(f"  Lost production:   {100 * lost_mean / total_mean:.1f}%")

    # ---- COMPONENTS ----
    print("\nCOMPONENT BREAKDOWN (direct cost, mean)")
    print("-" * 60)

    comp_stats = results["lifetime_component_stats"]

    for label, stats in comp_stats.items():
        share = 100 * stats["mean"] / total_mean if total_mean > 0 else 0

        print(f"  {label}:")
        print(f"    Mean:            {fmt(stats['mean'])} {currency}")
        print(f"    Share of total cost: {share:.1f}%")


    print("=" * 60)
    print("Marginal TOTAL O&M cost per 5-year period")
    print("(discounted to today, per MW)")
    print("=" * 60)

    for period, stats in results["marginal_total_stats"].items():
        start, _ = map(int, period.split("-"))
        display_year = 2030 + start

        if display_year == 2045:
            continue

        print(f"Year {display_year}")
        print("-" * 60)

        print(f"  CVaR 5% (best):   {fmt(stats['cvar5_low'] / results['total_production'])} {currency}/MW")
        print(f"  P5 (best 5%):     {fmt(stats['p5'] / results['total_production'])} {currency}/MW")
        print(f"  P25:              {fmt(stats['p25'] / results['total_production'])} {currency}/MW")
        print(f"  P50 (median):     {fmt(stats['p50'] / results['total_production'])} {currency}/MW")
        print(f"  P75:              {fmt(stats['p75'] / results['total_production'])} {currency}/MW")
        print(f"  P95 (worst 5%):   {fmt(stats['p95'] / results['total_production'])} {currency}/MW")
        print(f"  CVaR 95% (worst): {fmt(stats['cvar95_high'] / results['total_production'])} {currency}/MW")

        # --- Optional: show breakdown (mean only, properly aggregated) ---
        direct_mean = results["marginal_direct_stats"][period]["mean"]
        lost_mean   = results["marginal_lost_prod_stats"][period]["mean"]
        total_mean  = results["marginal_total_stats"][period]["mean"]

        if total_mean > 0:
            direct_share = 100 * direct_mean / total_mean
            lost_share   = 100 * lost_mean / total_mean

            print("\n  Breakdown (mean):")
            print(f"    Direct cost:         {direct_share:.1f}%")
            print(f"    Lost production:     {lost_share:.1f}%")

        print("=" * 60)




def results_to_excel_tables(results, regions, filename="om_results.xlsx"):

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:

        for region in regions:

            parkname = region.name
            park_results = results[parkname]

            data = {}

            for period, stats in park_results["marginal_stats"].items():
                start, end = map(int, period.split("-"))
                year = 2030 + start

                if year == 2045:
                    continue

                data[year] = {
                    "CVaR 5% (best)": stats["cvar5_low"] / park_results["total_production"],
                    "P5": stats["p5"] / park_results["total_production"],
                    "P25": stats["p25"] / park_results["total_production"],
                    "P50": stats["p50"] / park_results["total_production"],
                    "P75": stats["p75"] / park_results["total_production"],
                    "P95": stats["p95"] / park_results["total_production"],
                    "CVaR 95% (worst)": stats["cvar95_high"] / park_results["total_production"],
                }

            df = pd.DataFrame(data)
            df = df.sort_index(axis=1)
            df = df.transpose()  # years as rows, stats as columns

            # write region name in top-left cell
            df.to_excel(writer, sheet_name=parkname[:30])

            worksheet = writer.sheets[parkname[:30]]
            worksheet.cell(row=1, column=1).value = parkname





total_area_capacity_mw = 1000
capacity_per_turbine = 15
n_turbines = 20
number_of_parks = total_area_capacity_mw/(capacity_per_turbine*n_turbines)


OFFSHORE_FAILURE_MULTIPLIER = 1.26

# Instantiate WindRegion objects for each region
# Name, offshore failure multiplier, capacity factor, cost multiplier
Nordavind = WindRegion("Nordavind", OFFSHORE_FAILURE_MULTIPLIER, 0.494, 1.8)
Nordvest = WindRegion("Nordvest", OFFSHORE_FAILURE_MULTIPLIER, 0.483, 1.8)
Vestavind1 = WindRegion("Vestavind1", OFFSHORE_FAILURE_MULTIPLIER, 0.489, 1.8)
Vestavind2 = WindRegion("Vestavind2", OFFSHORE_FAILURE_MULTIPLIER, 0.512, 1.8)
SorvestA = WindRegion("SorvestA", OFFSHORE_FAILURE_MULTIPLIER, 0.545, 1.5)
SorvestB = WindRegion("SorvestB", OFFSHORE_FAILURE_MULTIPLIER, 0.543, 1.5)
SorvestC = WindRegion("SorvestC", OFFSHORE_FAILURE_MULTIPLIER, 0.551, 1.5)
SorvestD = WindRegion("SorvestD", OFFSHORE_FAILURE_MULTIPLIER, 0.545, 1.5)
SorvestE = WindRegion("SorvestE", OFFSHORE_FAILURE_MULTIPLIER, 0.561, 1.5)
SorvestF = WindRegion("SorvestF", OFFSHORE_FAILURE_MULTIPLIER, 0.559, 1.5)
Sonnavind = WindRegion("Sonnavind", OFFSHORE_FAILURE_MULTIPLIER, 0.565, 1.8)

WindRegions = [Nordavind, Nordvest, Vestavind1, Vestavind2, SorvestA, SorvestB, SorvestC, SorvestD, SorvestE, SorvestF, Sonnavind]

all_results = {}

for region in WindRegions:
    print(f"Simulating for region: {region.name}")
    result = simulate_wind_farm_OandM(n_turbines=n_turbines, capacity_per_turbine=capacity_per_turbine, horizon_years=25.0,
                                  number_of_parks=number_of_parks, capacity_factor=region.capacity_factor, price_per_mwh=1.15, discount_rate=0.07, 
                                  n_simulations=5000, random_seed=123, offshore_factor=region.offshore_multiplyer, cost_multiplier=region.cost_multiplyer)
    pretty_print_results(result, currency="kNOK")

    all_results[region.name] = result

    
#results_to_excel_tables(all_results, WindRegions, filename="om_results.xlsx")
