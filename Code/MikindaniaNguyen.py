from dataclasses import dataclass
from locale import currency
import pandas as pd
import numpy as np

kNOKEUR = 10.39/1000
kNOKGBP = 11.385/1000 #2016 average

target_years = [2030, 2035, 2040, 2050]
# Fixed trend multiplyer relative to 2030 to 2035, 2040 and 2050
fixed_trend_multiplyer = [0.9394, 0.8945, 0.8297] 
# Floating trend multiplyer relative to 2030 to 2035, 2040 and 2050
floating_trend_multiplyer = [0.815, 0.7012, 0.6334]

def read_market_price_file(filepath):
    """
    Reads a market price file in tab-separated format and converts it to a nested dictionary.
    
    Expected input format:
    Season | Region | Average PV (ore/kWh) | Period
    Fall   | O_Vestavind1 | 192.3148551 | 2030
    ...
    
    Output format:
    {
        "Vestavind1": {
            "2030": {
                "Fall": 192.3148551,
                "Summer": 172.6992945,
                ...
            },
            ...
        },
        ...
    }
    
    Args:
        filepath (str): Path to the market price file
        
    Returns:
        dict: Nested dictionary with structure: region -> year -> season -> price
    """
    # Read the file as tab-separated
    df = pd.read_csv(filepath)
    
    # Initialize the result dictionary
    market_price = {}
    
    # Process each row
    for _, row in df.iterrows():
        season = row['Season']
        region = row['Region']
        price = row['Average PV (ore/kWh)']
        period = str(row['Period'])

        # Initialize nested structure if needed
        if region not in market_price:
            market_price[region] = {}
        if period not in market_price[region]:
            market_price[region][period] = {}
        
        # Store the price
        market_price[region][period][season] = price
    
    return market_price


def convert_ore_kwh_to_knok_mwh(price_ore_kwh):
    """
    Convert price from øre/kWh to kNOK/MWh.
    
    Conversion factors:
    - 1 øre = 0.01 NOK
    - 1 kNOK = 1000 NOK
    - 1 MWh = 1000 kWh
    
    So: X øre/kWh * 0.01 NOK/øre * 1000 kWh/MWh / 1000 NOK/kNOK = X * 0.01 kNOK/MWh
    
    Args:
        price_ore_kwh (float): Price in øre/kWh
        
    Returns:
        float: Price in kNOK/MWh
    """
    return price_ore_kwh * 0.01


def get_season_from_time(t):
    """
    Determine the season based on the decimal part of time t.
    
    Season mapping:
    - Spring: 0.00 to 0.24
    - Summer: 0.25 to 0.49
    - Fall:   0.50 to 0.74
    - Winter: 0.75 to 0.99
    
    Args:
        t (float): Time value (e.g., 2.3 for t=2.3)
        
    Returns:
        str: Season name ('Spring', 'Summer', 'Fall', 'Winter')
    """
    decimal_part = t % 1.0
    
    if decimal_part < 0.25:
        return "Spring"
    elif decimal_part < 0.50:
        return "Summer"
    elif decimal_part < 0.75:
        return "Fall"
    else:
        return "Winter"


def get_market_year_from_time(t):
    """
    Determine the market year based on the integer part of time t.
    
    Year mapping (0-year basis):
    - 0-4 years → 2030
    - 5-9 years → 2035
    - 10-14 years → 2040
    - etc.
    
    Args:
        t (float): Time value (e.g., 2.3 for t=2.3)
        
    Returns:
        str: Year string (e.g., '2030', '2035')
    """
    integer_part = int(t)
    year_offset = (integer_part // 5) * 5
    base_year = 2030
    return str(base_year + year_offset)


def get_market_price(t, region_name, market_price_dict):
    """
    Get the market price for a given time and region.
    
    Args:
        t (float): Time value (e.g., 2.3 for t=2.3)
        region_name (str): Region name (e.g., 'Vestavind1', 'Nordavind')
        market_price_dict (dict): Market price dictionary with structure:
                                  region -> year -> season -> price (in øre/kWh)
        
    Returns:
        float: Market price in kNOK/MWh, or None if not found
    """
    season = get_season_from_time(t)
    year = get_market_year_from_time(t)
    
    try:
        price_ore_kwh = market_price_dict[region_name][year][season]
        return convert_ore_kwh_to_knok_mwh(price_ore_kwh)
    except KeyError:
        # Return None if the specific market data is not available
        return None


SEVERITIES = ["minor", "major", "replace"]

component_data = {
    "Generator": {
        "shape": 1.52,
        "scale": 18.72,
        "severity_probs": [0.5383, 0.3563, 0.1054],  # minor, major, replace

        "repair_time": {"minor": 7, "major": 24, "replace": 81},
        "material_cost": {"minor": 2667 * kNOKGBP, "major": 58333 * kNOKGBP, "replace": 1000000 * kNOKGBP},
        "technicians": {"minor": 2.2, "major": 2.7, "replace": 7.9}
    },

    "Gearbox": {
        "shape": 1.38,
        "scale": 15.02,
        "severity_probs": [0.6729, 0.0647, 0.2624],

        "repair_time": {"minor": 8, "major": 22, "replace": 231},
        "material_cost": {"minor": 380 * kNOKGBP, "major": 7609 * kNOKGBP, "replace": 700000 * kNOKGBP},
        "technicians": {"minor": 2.2, "major": 3.2, "replace": 17.2}
    },

    "Blades": {
        "shape": 0.75,
        "scale": 86.8,
        "severity_probs": [0.9764, 0.0214, 0.0022],

        "repair_time": {"minor": 9, "major": 21, "replace": 288},
        "material_cost": {"minor": 819 * kNOKGBP, "major": 7222 * kNOKGBP, "replace": 433333 * kNOKGBP},
        "technicians": {"minor": 2.1, "major": 3.3, "replace": 21}
    }
}

# Veseel data for offshore operations
vessel_data = {
    # Minor repairs correspond sto Crew Transfer Vessel (CTV) - Carrol
    "minor": {   # CTV
        "day_rate": 3000 *kNOKEUR,
        "mobilisation_cost": 0,
        "mobilisation_hours": 0,
        "speed_kmh": 37.04
    },
    # Major repairs correspond to Fast Support Vessel (FSV) - Carrol
    "major": {   # FSV
        "day_rate": 12500 * kNOKEUR,
        "mobilisation_cost": 0,
        "mobilisation_hours": 504,
        "speed_kmh": 22.224
    },
    # Major replacements correspond to Heavy Lift Vessel (HLV) - Carrol
    "replace": {  # HLV
        "day_rate": 150000 * kNOKEUR,
        "mobilisation_cost": 2100000 * kNOKEUR,
        "mobilisation_hours": 1440,
        "speed_kmh": 20.372
    }
}

tech_market_data = read_market_price_file("C:\\Users\\IFE13253\\OneDrive - Institutt for Energiteknikk\\Documents\\OffshoreRisk\\RiskSimulation\\MonteCarlo-PostProcces\\Results\\tech onshore shadow price.csv")
inc_market_data = read_market_price_file("C:\\Users\\IFE13253\\OneDrive - Institutt for Energiteknikk\\Documents\\OffshoreRisk\\RiskSimulation\\MonteCarlo-PostProcces\\Results\\inc onshore shadow price.csv")


# ComponentType(name="Blades", shape=0.75, scale=86.8,  down_time=147.0, direct_cost=375689 * cost_multiplier * kNOKEUR),
# ComponentType(name="Gearbox", shape=1.38, scale=15.02, down_time=261.0, direct_cost=647101 * cost_multiplier * kNOKEUR),
# ComponentType(name="Generator", shape=1.52, scale=18.72, down_time=126.0, direct_cost=232634 * cost_multiplier * kNOKEUR),

@dataclass
class ComponentType:
    """
    Data structure for a wind turbine component's reliability and cost parameters.
    """
    name: str           # Name of the component (e.g., 'Gearbox')
    shape: float        # Weibull shape parameter (k)
    scale: float        # Weibull scale parameter (λ) in years (characteristic life)
    severity_probs: list  # List of probabilities for each severity level (minor, major, replace)
    material_cost: dict  # Material cost for repair/replacement (in currency units)
    repair_time: dict  # Average repair time (in hours)
    nr_workers: dict     # Number of workers required for repair/replacement
    # Additional fields (e.g., variable maintenance cost, cost distributions) can be added if needed.

@dataclass
class VesselType:
    """
    Data structure for a vessel's operational parameters for offshore wind O&M.
    """
    name: str           # Name of the vessel type (e.g., 'CTV', 'FSV', 'HLV')
    day_rate: float     # Daily charter rate (in currency units)
    mobilisation_cost: float  # Mobilisation cost (in currency units)
    mobilisation_hours: float  # Mobilisation time (in hours)
    speed_kmh: float     # Average transit speed (in km/h)

@dataclass
class WindRegion:
    name: str 
    offshore_multiplyer: float
    capacity_factor: float
    distance_to_shore_km: float
    floating: bool


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
    capacity_per_turbine: float = 15.0,
    capacity_factor: float = 0.5,
    price_per_mwh: float = 125.11 *kNOKEUR,
    distance_to_shore_km: float = 50.0,
    daily_rate: float = 10000.0,
    discount_rate: float = 0.07,
    n_simulations: int = 10000,
    random_seed: int = None,
    offshore_factor: float = 1.0,
    shift_hours: float = 12,
    market_price_dict: dict = None,
    region_name: str = None
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

    cost_ratio = capacity_per_turbine/10 #Is the linear cost increase/decrese from the 10MW base cost estimation

    # Use default component types (key offshore turbine components) if none provided
    if component_types is None:
        component_types = [
            ComponentType(
                name="Blades",
                shape=component_data["Blades"]["shape"],
                scale=component_data["Blades"]["scale"],
                severity_probs=component_data["Blades"]["severity_probs"],
                material_cost={
                    k: cost_ratio * v
                    for k, v in component_data["Blades"]["material_cost"].items()
                },
                repair_time=component_data["Blades"]["repair_time"],
                nr_workers=component_data["Blades"]["technicians"]
            ),
            ComponentType(
                name="Gearbox",
                shape=component_data["Gearbox"]["shape"],
                scale=component_data["Gearbox"]["scale"],
                severity_probs=component_data["Gearbox"]["severity_probs"],
                material_cost={
                    k: cost_ratio * v
                    for k, v in component_data["Gearbox"]["material_cost"].items()
                },
                repair_time=component_data["Gearbox"]["repair_time"],
                nr_workers=component_data["Gearbox"]["technicians"]
            ),
            ComponentType(
                name="Generator",
                shape=component_data["Generator"]["shape"],
                scale=component_data["Generator"]["scale"],
                severity_probs=component_data["Generator"]["severity_probs"],
                material_cost={
                    k: cost_ratio * v
                    for k, v in component_data["Generator"]["material_cost"].items()
                },
                repair_time=component_data["Generator"]["repair_time"],
                nr_workers=component_data["Generator"]["technicians"]
            )
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
    total_eq_annual_costs_pv = np.zeros(n_simulations)
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

    CRF = discount_rate * (1 + discount_rate) ** horizon_years / ((1 + discount_rate) ** horizon_years - 1)
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

                    severity = rng.choice(SEVERITIES, p=comp.severity_probs)
                    
                    repair_time = comp.repair_time[severity]
                    
                    material_cost = comp.material_cost[severity]
                    
                    technicians = comp.nr_workers[severity]
                    
                    vessel = vessel_data[severity]

                    transit_hours = 2.0 * distance_to_shore_km / vessel["speed_kmh"]  # Round trip transit time in hours
                    working_days = np.ceil((transit_hours + repair_time) / shift_hours)

                    working_hours = transit_hours / 2 + repair_time
                    full_days = working_hours // shift_hours 
                    remaining_hours = working_hours % shift_hours

                    time_to_operation = full_days * 24 + remaining_hours

                    downtime_hours = vessel["mobilisation_hours"] + time_to_operation  # Total downtime for this failure event

                    labor_cost = technicians * working_days * daily_rate

                    transport_cost = vessel["mobilisation_cost"] + vessel["day_rate"] * working_days

                    direct_cost = material_cost + labor_cost + transport_cost

                    lost_mwh = downtime_hours * capacity_per_turbine * capacity_factor  # PLC MWh of production lost
                    
                    # Use dynamic market price if available and enabled, otherwise use fixed price
                    if market_price_dict is not None and region_name is not None:
                        market_price = get_market_price(t_fail, region_name, market_price_dict)
                        if market_price is not None:
                            price_to_use = market_price
                        else:
                            price_to_use = price_per_mwh
                    else:
                        price_to_use = price_per_mwh
                    
                    lost_prod_cost = lost_mwh * price_to_use
                    direct_cost_pv = direct_cost * pv_factor
                    lost_prod_cost_pv = lost_prod_cost * pv_factor
                    event_cost_pv = direct_cost_pv + lost_prod_cost_pv
                    equivalent_annual_cost = event_cost_pv * CRF

                    # store richer event tuple
                    events_pv.append(
                        (t_fail, comp_label, direct_cost_pv, lost_prod_cost_pv, event_cost_pv, equivalent_annual_cost)
                    )
        # Sort events by failure time for consistent accumulation
        events_pv.sort(key=lambda e: e[0])
        # Accumulate present-value costs up to each target offset for this run
        cum_sum_pv = 0.0
        cum_direct_pv = 0.0
        cum_lost_prod_pv = 0.0
        cum_sum_eq_annual = 0.0
        cum_component_pv = {label: 0.0 for label in component_labels}

        offset_idx = 0

        for t_fail, comp_label, direct_pv, lost_prod_pv, total_pv, eq_annual_cost in events_pv:
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
            cum_sum_eq_annual += eq_annual_cost
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
        total_eq_annual_costs_pv[sim] = cum_sum_eq_annual
        total_direct_costs_pv[sim] = cum_direct_pv
        total_lost_prod_costs_pv[sim] = cum_lost_prod_pv

        for label in component_labels:
            total_component_costs_pv[label][sim] = cum_component_pv[label]

    # Costs are normalized per MW later; no separate park-count scaling is applied.
    # Compute marginal 5-year period costs by differencing cumulative costs between successive target offsets
    marginal_direct_costs_pv_by_period = {}
    marginal_lost_prod_costs_pv_by_period = {}
    marginal_component_costs_pv_by_period = {}
    marginal_eq_annual_costs_pv_by_period = {}

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
        "n_turbines": n_turbines,
        "capacity_per_turbine": capacity_per_turbine,
        "installed_capacity": n_turbines * capacity_per_turbine,

        "lifetime_total_costs_pv": total_costs_pv,
        "lifetime_direct_costs_pv": total_direct_costs_pv,
        "lifetime_lost_prod_costs_pv": total_lost_prod_costs_pv,
        "lifetime_component_costs_pv": total_component_costs_pv,

        "lifetime_total_stats": summary_stats(total_costs_pv),
        "lifetime_direct_stats": summary_stats(total_direct_costs_pv),
        "lifetime_lost_prod_stats": summary_stats(total_lost_prod_costs_pv),
        "equivalent_annual_costs": summary_stats(total_eq_annual_costs_pv),

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


def pretty_print_results(results, currency="kNOK", floating=False):
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
    print(" Offshore Wind O&M Cost Risk Summary (Advanced cost model)")
    print("=" * 60)

    print(f"Number of turbines in park:    {fmt(results['n_turbines'])}")
    print(f"Capacity per turbine:          {fmt(results['capacity_per_turbine'])}")
    print(f"Installed capacity:            {fmt(results['installed_capacity'])} MW")
    print("=" * 60)
    print("Lifetime O&M Cost Breakdown (PV, discounted to today)")
    print("=" * 60)

    capacity = results["installed_capacity"]

    total_stats = results["lifetime_total_stats"]
    direct_stats = results["lifetime_direct_stats"]
    lost_stats = results["lifetime_lost_prod_stats"]
    eq_annual_stats = results["equivalent_annual_costs"]

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

    multiplyer = 1.0
    for year in target_years:
        print(f"\nCumulative O&M cost distribution up to year {year} (present value, discounted to today)")
        print("-" * 60)
        if floating:
            if year == 2030:
                multiplyer = 1
            elif year == 2035:
                multiplyer = floating_trend_multiplyer[0]
            elif year == 2040:
                multiplyer = floating_trend_multiplyer[1]
            elif year == 2050:
                multiplyer = floating_trend_multiplyer[2]
        else:
            if year == 2030:
                multiplyer = 1
            elif year == 2035:
                multiplyer = fixed_trend_multiplyer[0]
            elif year == 2040:
                multiplyer = fixed_trend_multiplyer[1]
            elif year == 2050:
                multiplyer = fixed_trend_multiplyer[2]


        print("=" * 60)
        print(f"O&M cost distribution normalized per MW for year {year} (present value, discounted to today)")
        print("-" * 60)
        print(f"  cvar5 (best 5%):   {fmt(multiplyer * total_stats['cvar5_low'] / capacity)} {currency}/MW")
        print(f"  p5 (best 5%):      {fmt(multiplyer * total_stats['p5'] / capacity)} {currency}/MW")
        print(f"  p25:               {fmt(multiplyer * total_stats['p25'] / capacity)} {currency}/MW")
        print(f"  p50 (median):      {fmt(multiplyer * total_stats['p50'] / capacity)} {currency}/MW")
        print(f"  p75:               {fmt(multiplyer * total_stats['p75'] / capacity )} {currency}/MW")
        print(f"  p95 (worst 5%):    {fmt(multiplyer * total_stats['p95'] / capacity)} {currency}/MW")
        print(f"  cvar95 (worst 5%): {fmt(multiplyer * total_stats['cvar95_high'] / capacity)} {currency}/MW")
        print("=" * 60)
        print(f"Equivalent annual O&M cost for year {year} (PV, discounted to today)")
        print("-" * 60)
        print(f"  Eq cvar5 (best 5%):   {fmt(multiplyer * eq_annual_stats['cvar5_low'] / capacity)} {currency}/MW")
        print(f"  Eq p5 (best 5%):      {fmt(multiplyer * eq_annual_stats['p5'] / capacity)} {currency}/MW")
        print(f"  Eq p25:               {fmt(multiplyer * eq_annual_stats['p25'] / capacity)} {currency}/MW")
        print(f"  Eq p50 (median):      {fmt(multiplyer * eq_annual_stats['p50'] / capacity)} {currency}/MW")
        print(f"  Eq p75:               {fmt(multiplyer * eq_annual_stats['p75'] / capacity)} {currency}/MW")
        print(f"  Eq p95 (worst 5%):    {fmt(multiplyer * eq_annual_stats['p95'] / capacity)} {currency}/MW")
        print(f"  Eq cvar95 (worst 5%): {fmt(multiplyer * eq_annual_stats['cvar95_high'] / capacity)} {currency}/MW")
        print("=" * 60)


    # print("Marginal TOTAL O&M cost per 5-year period")
    # print("(discounted to today, per MW)")
    # print("=" * 60)

    # for period, stats in results["marginal_total_stats"].items():
    #     start, _ = map(int, period.split("-"))
    #     display_year = 2030 + start

    #     if display_year == 2045:
    #         continue

    #     print(f"Year {display_year}")
    #     print("-" * 60)

    #     print(f"  CVaR 5% (best):   {fmt(stats['cvar5_low'] / results['installed_capacity'])} {currency}/MW")
    #     print(f"  P5 (best 5%):     {fmt(stats['p5'] / results['installed_capacity'])} {currency}/MW")
    #     print(f"  P25:              {fmt(stats['p25'] / results['installed_capacity'])} {currency}/MW")
    #     print(f"  P50 (median):     {fmt(stats['p50'] / results['installed_capacity'])} {currency}/MW")
    #     print(f"  P75:              {fmt(stats['p75'] / results['installed_capacity'])} {currency}/MW")
    #     print(f"  P95 (worst 5%):   {fmt(stats['p95'] / results['installed_capacity'])} {currency}/MW")
    #     print(f"  CVaR 95% (worst): {fmt(stats['cvar95_high'] / results['installed_capacity'])} {currency}/MW")

    #     # --- Optional: show breakdown (mean only, properly aggregated) ---
    #     direct_mean = results["marginal_direct_stats"][period]["mean"]
    #     lost_mean   = results["marginal_lost_prod_stats"][period]["mean"]
    #     total_mean  = results["marginal_total_stats"][period]["mean"]

    #     if total_mean > 0:
    #         direct_share = 100 * direct_mean / total_mean
    #         lost_share   = 100 * lost_mean / total_mean

    #         print("\n  Breakdown (mean):")
    #         print(f"    Direct cost:         {direct_share:.1f}%")
    #         print(f"    Lost production:     {lost_share:.1f}%")

    #     print("=" * 60)




def results_to_excel_tables(results, regions, floating=False, filename="om_results.xlsx"):
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:

        for region in regions:

            parkname = region.name
            park_results = results[parkname]

            eq_annual_stats = park_results["equivalent_annual_costs"]
            capacity = park_results["installed_capacity"]

            data = {}

            multiplyer = 1.0
            for year in target_years:
                if floating:
                    if year == 2030:
                        multiplyer = 1
                    elif year == 2035:
                        multiplyer = floating_trend_multiplyer[0]
                    elif year == 2040:
                        multiplyer = floating_trend_multiplyer[1]
                    elif year == 2050:
                        multiplyer = floating_trend_multiplyer[2]
                else:
                    if year == 2030:
                        multiplyer = 1
                    elif year == 2035:
                        multiplyer = fixed_trend_multiplyer[0]
                    elif year == 2040:
                        multiplyer = fixed_trend_multiplyer[1]
                    elif year == 2050:
                        multiplyer = fixed_trend_multiplyer[2]

                data[year] = {
                    "CVaR 5% (best)": multiplyer * eq_annual_stats["cvar5_low"] / capacity,
                    "P5": multiplyer * eq_annual_stats["p5"] / capacity,
                    "P25": multiplyer * eq_annual_stats["p25"] / capacity,
                    "P50": multiplyer * eq_annual_stats["p50"] / capacity,
                    "P75": multiplyer * eq_annual_stats["p75"] / capacity,
                    "P95": multiplyer * eq_annual_stats["p95"] / capacity,
                    "CVaR 95% (worst)": multiplyer * eq_annual_stats["cvar95_high"] / capacity,
                }

            df = pd.DataFrame(data)
            df = df.sort_index(axis=1)
            df = df.transpose()  # years as rows, stats as columns

            # write region name in top-left cell
            df.to_excel(writer, sheet_name=parkname[:30])

            worksheet = writer.sheets[parkname[:30]]
            worksheet.cell(row=1, column=1).value = parkname





capacity_per_turbine = 15
n_turbines = 20

OFFSHORE_FAILURE_MULTIPLIER = 1.27

# Instantiate WindRegion objects for each region
# Name, offshore failure multiplier, capacity factor, distance from shore (km)
Nordavind = WindRegion("Nordavind", OFFSHORE_FAILURE_MULTIPLIER, 0.494, 200, floating=True)
Nordvest = WindRegion("Nordvest", OFFSHORE_FAILURE_MULTIPLIER, 0.483, 110, floating=True)
Vestavind1 = WindRegion("Vestavind1", OFFSHORE_FAILURE_MULTIPLIER, 0.489, 70, floating=True)
Vestavind2 = WindRegion("Vestavind2", OFFSHORE_FAILURE_MULTIPLIER, 0.512, 50, floating=True)
SorvestA = WindRegion("SorvestA", OFFSHORE_FAILURE_MULTIPLIER, 0.545, 122, floating=False)
SorvestB = WindRegion("SorvestB", OFFSHORE_FAILURE_MULTIPLIER, 0.543, 152, floating=False)
SorvestC = WindRegion("SorvestC", OFFSHORE_FAILURE_MULTIPLIER, 0.551, 153, floating=False)
SorvestD = WindRegion("SorvestD", OFFSHORE_FAILURE_MULTIPLIER, 0.545, 221, floating=False)
SorvestE = WindRegion("SorvestE", OFFSHORE_FAILURE_MULTIPLIER, 0.561, 112, floating=False)
SorvestF = WindRegion("SorvestF", OFFSHORE_FAILURE_MULTIPLIER, 0.559, 152, floating=False)
Sonnavind = WindRegion("Sonnavind", OFFSHORE_FAILURE_MULTIPLIER, 0.565, 60, floating=True)

WindRegions = [Nordavind, Nordvest, Vestavind1, Vestavind2, SorvestA, SorvestB, SorvestC, SorvestD, SorvestE, SorvestF, Sonnavind]

# Toggle between market prices and fixed subsidy price
USE_MARKET_PRICES = True  # Set to False to use fixed subsidy price (1.15 kNOK/MWh)
subsidy_price = 1.15  # Fixed subsidy price in currency units per MWh
scenario = ["TECH", "INC"]

all_results = {}


if USE_MARKET_PRICES:
    for s in scenario:
        if s == "TECH":
            print("Running TECH scenario...")
            market_data = tech_market_data
            filename = "om_results_tech.xlsx"
        elif s == "INC":
            print("Running INC scenario...")
            market_data = inc_market_data 
            filename = "om_results_inc.xlsx" 

        for region in WindRegions:
            print(f"Simulating for region: {region.name}")
            
            # Map Sorvest sub-regions to the single "Sorvest" market region
            market_region_name = region.name
            if region.name.startswith("Sorvest") or region.name =="Sonnavind" or region.name == "Vestavind2":
                market_region_name = "NO2"
            elif region.name == "Nordvest":
                market_region_name = "NO3"
            elif region.name == "Nordavind":
                market_region_name = "NO4"
            elif region.name == "Vestavind1":
                market_region_name = "NO5"
            
            result = simulate_wind_farm_OandM(n_turbines=n_turbines, capacity_per_turbine=capacity_per_turbine, horizon_years=25.0,
                                        capacity_factor=region.capacity_factor, price_per_mwh=subsidy_price, discount_rate=0.07, 
                                        distance_to_shore_km=region.distance_to_shore_km, daily_rate =10, n_simulations=5000, random_seed=123, offshore_factor=region.offshore_multiplyer,
                                        market_price_dict=market_data, region_name=market_region_name)
            pretty_print_results(result, currency="kNOK", floating=region.floating)

            all_results[region.name] = result

            
        results_to_excel_tables(all_results, WindRegions, floating=region.floating, filename=filename) 
else:
    print("Running fixed subsidy price scenario...")
    market_data = None  # Not used when fixed price is applied
    filename = "om_results_subsidies.xlsx"

    for region in WindRegions:
        print(f"Simulating for region: {region.name}")
        
        # Map Sorvest sub-regions to the single "Sorvest" market region
        market_region_name = region.name
        if region.name.startswith("Sorvest") or region.name =="Sonnavind" or region.name == "Vestavind2":
            market_region_name = "NO2"
        elif region.name == "Nordvest":
            market_region_name = "NO3"
        elif region.name == "Nordavind":
            market_region_name = "NO4"
        elif region.name == "Vestavind1":
            market_region_name = "NO5"

        result = simulate_wind_farm_OandM(n_turbines=n_turbines, capacity_per_turbine=capacity_per_turbine, horizon_years=25.0,
                                    capacity_factor=region.capacity_factor, price_per_mwh=subsidy_price, discount_rate=0.07, 
                                    distance_to_shore_km=region.distance_to_shore_km, daily_rate =10, n_simulations=5000, random_seed=123, offshore_factor=region.offshore_multiplyer,
                                    market_price_dict=market_data, region_name=market_region_name)
        pretty_print_results(result, currency="kNOK", floating=region.floating)

        all_results[region.name] = result

        
    results_to_excel_tables(all_results, WindRegions, floating=region.floating, filename=filename)

