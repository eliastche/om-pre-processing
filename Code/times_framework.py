from dataclasses import dataclass
import pandas as pd
import numpy as np

kNOKEUR = 10.39/1000
kNOKGBP = 11.385/1000 #2016 average

# Fixed trend multiplyer relative to 2030 to 2035, 2040 and 2050
fixed_trend_multiplyer = [0.9394, 0.8945, 0.8297] 
# Floating trend multiplyer relative to 2030 to 2035, 2040 and 2050
floating_trend_multiplyer = [0.815, 0.7012, 0.6334]

def get_cost_trend_multiplier(calendar_year: int, floating: bool = True):
    """
    Return O&M direct-cost trend multiplier relative to 2030.

    Existing input data:
    - 2030: 1.0
    - 2035, 2040, 2050: from fixed/floating trend multiplier lists

    For 2045, use linear interpolation between 2040 and 2050.
    For years beyond 2050, keep the 2050 value constant.
    """

    if floating:
        trend_points = {
            2030: 1.0,
            2035: floating_trend_multiplyer[0],
            2040: floating_trend_multiplyer[1],
            2050: floating_trend_multiplyer[2],
        }
    else:
        trend_points = {
            2030: 1.0,
            2035: fixed_trend_multiplyer[0],
            2040: fixed_trend_multiplyer[1],
            2050: fixed_trend_multiplyer[2],
        }

    # Exact value available
    if calendar_year in trend_points:
        return trend_points[calendar_year]

    # Before 2030: keep 2030 cost level
    if calendar_year <= 2030:
        return trend_points[2030]

    # Between 2040 and 2050, interpolate, e.g. 2045
    if 2040 < calendar_year < 2050:
        y0, y1 = 2040, 2050
        m0, m1 = trend_points[y0], trend_points[y1]
        return m0 + (m1 - m0) * ((calendar_year - y0) / (y1 - y0))

    # Between 2035 and 2040, interpolate if ever needed
    if 2035 < calendar_year < 2040:
        y0, y1 = 2035, 2040
        m0, m1 = trend_points[y0], trend_points[y1]
        return m0 + (m1 - m0) * ((calendar_year - y0) / (y1 - y0))

    # Between 2030 and 2035, interpolate if ever needed
    if 2030 < calendar_year < 2035:
        y0, y1 = 2030, 2035
        m0, m1 = trend_points[y0], trend_points[y1]
        return m0 + (m1 - m0) * ((calendar_year - y0) / (y1 - y0))

    # Beyond 2050: hold 2050 value constant
    if calendar_year >= 2050:
        return trend_points[2050]

    return 1.0

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
        
        # Extract region name from format "O_Vestavind1" -> "Vestavind1"
        region_name = region.split('_')[1]  # Remove O_ prefix
        
        # Initialize nested structure if needed
        if region_name not in market_price:
            market_price[region_name] = {}
        if period not in market_price[region_name]:
            market_price[region_name][period] = {}
        
        # Store the price
        market_price[region_name][period][season] = price
    
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


def get_market_year_from_time(t, start_year=2030):
    """
    Return market year as integer.
    Example:
    age 0-4   -> start_year
    age 5-9   -> start_year + 5
    age 10-14 -> start_year + 10
    """
    integer_part = int(t)
    year_offset = (integer_part // 5) * 5
    return start_year + year_offset


def get_market_price(t, region_name, market_price_dict, start_year=2030):
    """
    Get market price in kNOK/MWh.
    Years beyond 2050 use 2050 market price.
    Raises an error if data is missing.
    """
    season = get_season_from_time(t)

    year_int = get_market_year_from_time(t, start_year=start_year)

    if year_int > 2050:
        year_int = 2050

    year = str(year_int)

    if region_name not in market_price_dict:
        raise KeyError(
            f"Region '{region_name}' not found in market price data. "
            f"Available regions: {list(market_price_dict.keys())}"
        )

    if year not in market_price_dict[region_name]:
        raise KeyError(
            f"Year '{year}' not found for region '{region_name}'. "
            f"Available years: {list(market_price_dict[region_name].keys())}"
        )

    if season not in market_price_dict[region_name][year]:
        raise KeyError(
            f"Season '{season}' not found for region '{region_name}', year '{year}'. "
            f"Available seasons: {list(market_price_dict[region_name][year].keys())}"
        )

    price_ore_kwh = market_price_dict[region_name][year][season]
    return convert_ore_kwh_to_knok_mwh(price_ore_kwh)

def get_required_market_years(start_year: int, lifetime: int = 25, max_market_year: int = 2050):
    """
    Market years needed for a project starting in start_year.

    Example:
    start_year=2030, lifetime=25 -> [2030, 2035, 2040, 2045, 2050]
    start_year=2035, lifetime=25 -> [2035, 2040, 2045, 2050]
    start_year=2050, lifetime=25 -> [2050]

    Years beyond max_market_year are assumed to use max_market_year.
    """
    required_years = []

    for offset in range(0, lifetime, 5):
        year = start_year + offset

        if year > max_market_year:
            year = max_market_year

        if year not in required_years:
            required_years.append(year)

    return required_years

def market_data_available_for_project(
    market_data: dict,
    region_name: str,
    start_year: int,
    lifetime: int = 25,
    required_seasons=("Spring", "Summer", "Fall", "Winter"),
    max_market_year: int = 2050,
):
    """
    Returns True if a region has all required market-price data for a project.

    If False, the investment candidate should be skipped entirely.
    """

    if region_name not in market_data:
        return False, f"region '{region_name}' not found"

    required_years = get_required_market_years(
        start_year=start_year,
        lifetime=lifetime,
        max_market_year=max_market_year
    )

    for year_int in required_years:
        year = str(year_int)

        if year not in market_data[region_name]:
            return False, f"year '{year}' missing for region '{region_name}'"

        missing_seasons = [
            season for season in required_seasons
            if season not in market_data[region_name][year]
        ]

        if missing_seasons:
            return False, (
                f"missing seasons {missing_seasons} for region '{region_name}', year '{year}'"
            )

    return True, "market data available"

SEVERITIES = ["minor", "major", "replace"]

component_data = {
    "Generator": {
        "shape": 1.52,
        "scale": 18.72,
        "severity_probs": [0.5383, 0.3563, 0.1054],  # minor, major, replace

        "repair_time": {"minor": 7, "major": 24, "replace": 81},
        "material_cost": {"minor": 160 * kNOKGBP, "major": 3500 * kNOKGBP, "replace": 60000 * kNOKGBP},
        "technicians": {"minor": 2.2, "major": 2.7, "replace": 7.9}
    },

    "Gearbox": {
        "shape": 1.38,
        "scale": 15.02,
        "severity_probs": [0.6729, 0.0647, 0.2624],

        "repair_time": {"minor": 8, "major": 22, "replace": 231},
        "material_cost": {"minor": 125 * kNOKGBP, "major": 2500 * kNOKGBP, "replace": 230000 * kNOKGBP},
        "technicians": {"minor": 2.2, "major": 3.2, "replace": 17.2}
    },

    "Blades": {
        "shape": 0.75,
        "scale": 86.8,
        "severity_probs": [0.9764, 0.0214, 0.0022],

        "repair_time": {"minor": 9, "major": 21, "replace": 288},
        "material_cost": {"minor": 170 * kNOKGBP, "major": 1500 * kNOKGBP, "replace": 90000 * kNOKGBP},
        "technicians": {"minor": 2.1, "major": 3.3, "replace": 21}
    }
}

# Veseel data for offshore operations
vessel_data = {
    # Minor repairs correspond sto Crew Transfer Vessel (CTV) - Carrol
    "minor": {   # CTV
        "day_rate": 1750 *kNOKGBP,
        "mobilisation_cost": 0,
        "mobilisation_hours": 0,
        "speed_kmh": 37.04
    },
    # Major repairs correspond to Fast Support Vessel (FSV) - Carrol
    "major": {   # FSV
        "day_rate": 9500 * kNOKGBP,
        "mobilisation_cost": 0,
        "mobilisation_hours": 504,
        "speed_kmh": 22.224
    },
    # Major replacements correspond to Heavy Lift Vessel (HLV) - Carrol
    "replace": {  # HLV
        "day_rate": 150000 * kNOKGBP,
        "mobilisation_cost": 500000 * kNOKGBP,
        "mobilisation_hours": 1440,
        "speed_kmh": 20.372
    }
}

tech_market_data = read_market_price_file("C:\\Users\\IFE13253\\OneDrive - Institutt for Energiteknikk\\Documents\\OffshoreRisk\\RiskSimulation\\MonteCarlo-PostProcces\\Results\\Shadow power price by region- tech.csv")
inc_market_data = read_market_price_file("C:\\Users\\IFE13253\\OneDrive - Institutt for Energiteknikk\\Documents\\OffshoreRisk\\RiskSimulation\\MonteCarlo-PostProcces\\Results\\Shadow power price by region - inc.csv")


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
    floating: bool = True  # Default to True for floating wind farms, can be set to False for fixed-bottom


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
    start_year: int = 2030,
    lifetime: int = 25,
    floating: bool = True,
    number_of_parks: float = 1.0,
    capacity_per_turbine: float = 15.0,
    capacity_factor: float = 0.5,
    price_per_mwh: float = 125.11 *kNOKEUR,
    distance_to_shore_km: float = 50.0,
    daily_rate: float = 10000.0,
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

    horizon_years = lifetime

    if horizon_years < 0:
        raise ValueError("lifetime must be a positive integer representing the number of years to simulate.")
    # Use default component types (key offshore turbine components) if none provided
    if component_types is None:
        component_types = [
            ComponentType(
                name="Blades",
                shape=component_data["Blades"]["shape"],
                scale=component_data["Blades"]["scale"],
                severity_probs=component_data["Blades"]["severity_probs"],
                material_cost=component_data["Blades"]["material_cost"],
                repair_time=component_data["Blades"]["repair_time"],
                nr_workers=component_data["Blades"]["technicians"]
            ),
            ComponentType(
                name="Gearbox",
                shape=component_data["Gearbox"]["shape"],
                scale=component_data["Gearbox"]["scale"],
                severity_probs=component_data["Gearbox"]["severity_probs"],
                material_cost=component_data["Gearbox"]["material_cost"],
                repair_time=component_data["Gearbox"]["repair_time"],
                nr_workers=component_data["Gearbox"]["technicians"]
            ),
            ComponentType(
                name="Generator",
                shape=component_data["Generator"]["shape"],
                scale=component_data["Generator"]["scale"],
                severity_probs=component_data["Generator"]["severity_probs"],
                material_cost=component_data["Generator"]["material_cost"],
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
                        price_to_use = get_market_price(
                            t_fail,
                            region_name,
                            market_price_dict,
                            start_year=start_year
                        )
                    else:
                        price_to_use = price_per_mwh

                    calendar_year = int(get_market_year_from_time(t_fail, start_year=start_year))
                    cost_trend_multiplier = get_cost_trend_multiplier(calendar_year=calendar_year, floating=floating)
                    
                    lost_prod_cost = lost_mwh * price_to_use
                    direct_cost_pv = direct_cost * cost_trend_multiplier
                    lost_prod_cost_pv = lost_prod_cost * cost_trend_multiplier
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
        "total_yearly_capacity": number_of_parks * n_turbines * capacity_per_turbine, #* capacity_factor * horizon_years,

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


def pretty_print_results(results, start_year=2030, currency="kNOK", floating=True):
    """
    Nicely print summary statistics from the O&M Monte Carlo simulation.

    Parameters
    ----------
    results : dict
        Output dictionary from simulate_wind_farm_OandM()
    start_year : int
        Starting year for the simulation
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
    print(f"Total yearly capacity:         {fmt(results['total_yearly_capacity'])} MW")
    print("=" * 60)
    print("Lifetime O&M Cost Breakdown (PV, discounted to today)")
    print("(per MW)")
    print("=" * 60)

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
    print("Marginal TOTAL O&M cost (average per 5-year period)")
    print("(Adjusted with cost trend, per MW)")
    print("=" * 60)

    for period, stats in results["marginal_total_stats"].items():
        start, _ = map(int, period.split("-"))
        display_year = start_year + start

        if display_year > 2050 or display_year == 2045:
            continue  # skip years beyond 2050

        cost_multiplier = get_cost_trend_multiplier(calendar_year=display_year, floating=floating)

        print(f"Year {display_year} with cost trend multiplier: {cost_multiplier:.4f}")
        print("-" * 60)

        print(f"  CVaR 5% (best):   {fmt(stats['cvar5_low'] / (results['total_yearly_capacity'] * 5))} {currency}/MW")
        print(f"  P5 (best 5%):     {fmt(stats['p5'] / (results['total_yearly_capacity'] * 5))} {currency}/MW")
        print(f"  P25:              {fmt(stats['p25'] / (results['total_yearly_capacity'] * 5))} {currency}/MW")
        print(f"  P50 (median):     {fmt(stats['p50'] / (results['total_yearly_capacity'] * 5))} {currency}/MW")
        print(f"  P75:              {fmt(stats['p75'] / (results['total_yearly_capacity'] * 5))} {currency}/MW")
        print(f"  P95 (worst 5%):   {fmt(stats['p95'] / (results['total_yearly_capacity'] * 5))} {currency}/MW")
        print(f"  CVaR 95% (worst): {fmt(stats['cvar95_high'] / (results['total_yearly_capacity'] * 5))} {currency}/MW")

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

            current_row = 0

            for start_year, park_results in sorted(results[parkname].items()):

                data = {}

                for period, stats in park_results["marginal_total_stats"].items():

                    start, end = map(int, period.split("-"))
                    year = start_year + start

                    if year > 2050 or year == 2045:
                        continue

                    data[year] = {
                        "CVaR 5% (best)": stats["cvar5_low"] / (park_results["total_yearly_capacity"] * 5),
                        "P5": stats["p5"] / (park_results["total_yearly_capacity"] * 5),
                        "P25": stats["p25"] / (park_results["total_yearly_capacity"] * 5),
                        "P50": stats["p50"] / (park_results["total_yearly_capacity"] * 5),
                        "P75": stats["p75"] / (park_results["total_yearly_capacity"] * 5),
                        "P95": stats["p95"] / (park_results["total_yearly_capacity"] * 5),
                        "CVaR 95% (worst)": stats["cvar95_high"] / (park_results["total_yearly_capacity"] * 5),
                    }

                df = pd.DataFrame(data).T

                # Write title
                pd.DataFrame([[f"Start year {start_year}"]]).to_excel(
                    writer,
                    sheet_name=parkname[:30],
                    startrow=current_row,
                    index=False,
                    header=False
                )

                # Write table underneath
                df.to_excel(
                    writer,
                    sheet_name=parkname[:30],
                    startrow=current_row + 1
                )

                current_row += len(df) + 5



lifetime_years = 25
total_area_capacity_mw = 1000
capacity_per_turbine = 15
n_turbines = 20
number_of_parks = total_area_capacity_mw/(capacity_per_turbine*n_turbines)


OFFSHORE_FAILURE_MULTIPLIER = 1.26

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
USE_MARKET_PRICES = False  # Set to False to use fixed subsidy price (1.15 kNOK/MWh)

scenario = ["TECH"]

all_results = {}
start_years = [2030, 2035, 2040, 2050]
skipped_investments = []

for start_year in start_years:

    if USE_MARKET_PRICES:

        for s in scenario:

            if s == "TECH":
                print(f"Running TECH-{start_year} scenario...")
                market_data = tech_market_data
                filename = f"om_results_tech_{start_year}.xlsx"

            elif s == "INC":
                print(f"Running INC-{start_year} scenario...")
                market_data = inc_market_data
                filename = f"om_results_inc_{start_year}.xlsx"

            # Use nested structure so results do not overwrite each other
            if s not in all_results:
                all_results[s] = {}

            if start_year not in all_results[s]:
                all_results[s][start_year] = {}

            for region in WindRegions:
                print(f"Checking investment candidate: {s}-{start_year}-{region.name}")

                market_region_name = region.name

                # If you intentionally use Sorvest prices for all Sorvest sub-zones:
                if region.name.startswith("Sorvest"):
                    market_region_name = "Sorvest"

                # IMPORTANT:
                # Only keep this if you deliberately want Sonnavind to use Sorvest prices.
                # If Sonnavind should be unavailable when missing from market data,
                # then remove this line.
                if region.name == "Sonnavind":
                    market_region_name = "Sorvest"

                is_available, reason = market_data_available_for_project(
                    market_data=market_data,
                    region_name=market_region_name,
                    start_year=start_year,
                    lifetime=lifetime_years
                )

                if not is_available:
                    print(
                        f"Skipping {s}-{start_year}-{region.name}: "
                        f"no valid wind investment candidate because {reason}"
                    )

                    skipped_investments.append({
                        "scenario": s,
                        "start_year": start_year,
                        "region": region.name,
                        "market_region": market_region_name,
                        "reason": reason
                    })

                    continue

                print(f"Simulating for region: {region.name}")

                result = simulate_wind_farm_OandM(
                    n_turbines=n_turbines,
                    capacity_per_turbine=capacity_per_turbine,
                    start_year=start_year,
                    lifetime=lifetime_years,
                    floating=region.floating,
                    number_of_parks=number_of_parks,
                    capacity_factor=region.capacity_factor,
                    price_per_mwh=1.15,
                    distance_to_shore_km=region.distance_to_shore_km,
                    daily_rate=10,
                    n_simulations=10000,
                    random_seed=123,
                    offshore_factor=region.offshore_multiplyer,
                    market_price_dict=market_data,
                    region_name=market_region_name
                )

                # pretty_print_results(
                #     result,
                #     start_year=start_year,
                #     currency="kNOK",
                #     floating=region.floating
                # )

                # all_results[s][region.name][start_year] = result

                
            #results_to_excel_tables(all_results, WindRegions, start_year=start_year, filename=filename) 
    else:
        print(f"Running fixed subsidy price {start_year} scenario...")
        market_data = None  # Not used when fixed price is applied
        filename = f"om_results_subsidies_{start_year}.xlsx"

        for region in WindRegions:
            print(f"Simulating for region: {region.name}")
            
            # Map Sorvest sub-regions to the single "Sorvest" market region
            market_region_name = region.name
            if region.name.startswith("Sorvest") or region.name == "Sonnavind":
                market_region_name = "Sorvest"
            
            result = simulate_wind_farm_OandM(n_turbines=n_turbines, capacity_per_turbine=capacity_per_turbine, start_year=start_year, lifetime=lifetime_years, floating=region.floating,
                                        number_of_parks=number_of_parks, capacity_factor=region.capacity_factor, price_per_mwh=1.15, 
                                        distance_to_shore_km=region.distance_to_shore_km, daily_rate =10, n_simulations=10000, random_seed=123, offshore_factor=region.offshore_multiplyer,
                                        market_price_dict=market_data, region_name=market_region_name)
            pretty_print_results(result, start_year=start_year, currency="kNOK", floating=region.floating)

            if region.name not in all_results:
                all_results[region.name] = {}   

            all_results[region.name][start_year] = result

            
results_to_excel_tables(all_results, WindRegions, filename=filename)