from dataclasses import dataclass
import numpy as np

kNOKEUR = 10.39/1000
#offshore_factor = 1.26

#ComponentType(name="Blades", shape=0.75, scale=86.8 * 0.8,  repair_time=147.0, direct_cost=375689 * kNOKEUR),
#ComponentType(name="Gearbox", shape=1.38, scale=15.02 * 0.8, repair_time=261.0, direct_cost=647101 * kNOKEUR),
#ComponentType(name="Generator", shape=1.52, scale=18.72 * 0.8, repair_time=126.0, direct_cost=232634 * kNOKEUR),

@dataclass
class ComponentType:
    """Defines a wind turbine component's reliability and cost parameters for simulation."""
    name: str
    shape: float        # Weibull shape parameter (β)
    scale: float        # Weibull scale parameter (η, in years)
    is_major: bool      # True if failure requires heavy-lift vessel and major repair effort
    part_cost: float    # Replacement part cost (NOK)
    avg_repair_days: float  # Average downtime for repair (days, includes expected weather delay)
    crew_count: int     # Technicians required
    crew_days: float    # Total technician-days needed (crew_count * days if concurrent)
    use_heavy_vessel: bool  # True if a heavy-lift vessel (jack-up crane) is needed

@dataclass
class WindRegion:
    name: str
    distance_km: float
    foundation_type: str
    water_depth_m: float
    hvdc_connection: bool
    maturity: str
    capacity_factor: float



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
        dt = rng.weibull(component.shape) * component.scale #(component.scale/ (offshore_factor ** (1.0 / component.shape)))
        t += dt
        if t > horizon_years:
            break
        times.append(t)
    return times

def sample_downtime_days(component, rng):
    return rng.lognormal(mean=np.log(component.avg_repair_days), sigma=0.3)

def compute_vessel_cost(component, site, downtime_days, rng, hlv_daily_rate, hlv_mobilization):
    transit_days = 2 * site.distance_km / 20.0 / 24.0
    weather_delay = rng.lognormal(mean=0.2, sigma=0.5)

    if component.use_heavy_vessel:
        return hlv_mobilization + hlv_daily_rate * (
            downtime_days + transit_days + weather_delay
        )
    return 0.0


def compute_event_cost(
    component,
    site,
    t_fail,
    rng,
    capacity_per_turbine,
    capacity_factor,
    labor_rate,
    hlv_daily_rate,
    hlv_mobilization,
    cfd_active,
    cfd_strike_price,
    cfd_duration_years
):
    downtime_days = sample_downtime_days(component, rng)

    price = get_power_price(
        t_fail, rng,
        cfd_active,
        cfd_strike_price,
        cfd_duration_years
    )

    labor_cost = component.crew_count * component.crew_days * 24.0 * labor_rate

    vessel_cost = compute_vessel_cost(
        component, site, downtime_days, rng,
        hlv_daily_rate,
        hlv_mobilization
    )

    lost_mwh = capacity_per_turbine * capacity_factor * downtime_days * 24.0
    lost_prod_cost = lost_mwh * price

    return (
        component.part_cost +
        labor_cost +
        vessel_cost +
        lost_prod_cost
    )

def simulate_hvdc_failure(site, rng, horizon_years, total_cap, capacity_factor,
                         discount_rate, price_fn):

    events = []

    n_failures = rng.poisson(lam=0.05 * horizon_years)

    for _ in range(n_failures):
        t = rng.uniform(0, horizon_years)

        downtime_days = 30.0
        price = price_fn(t)

        lost_mwh = total_cap * capacity_factor * downtime_days * 24.0
        cost = 1000.0 + lost_mwh * price

        pv = (1 + discount_rate) ** -t
        events.append((t, cost * pv))

    return events

def get_seasonal_price(t, base_price):
    # season index
    season = int((t % 1) * 4)  # 0=winter,1=spring,2=summer,3=fall

    # --- 2030 values ---
    s2030 = [0.67, 0.56, 0.67, 0.73]
    avg2030 = np.mean(s2030)

    # --- 2050 values ---
    s2050 = [0.65, 0.41, 0.65, 0.98]
    avg2050 = np.mean(s2050)

    # convert to relative factors
    f2030 = [x / avg2030 for x in s2030]
    f2050 = [x / avg2050 for x in s2050]

    # time interpolation (0=2030, 25=2050)
    weight = min(t / 25.0, 1.0)

    factor = (1 - weight) * f2030[season] + weight * f2050[season]

    return base_price * factor


def get_power_price(t, rng, cfd_active, strike_price, cfd_years):
    
    # CfD (fixed nominal — recommended)
    if cfd_active and t <= cfd_years:
        return strike_price
    
    years = np.array([0, 5, 10, 20])
    prices = np.array([0.66, 0.76, 0.80, 0.67])

    base_price = np.interp(t, years, prices)
    season_price = get_seasonal_price(t, base_price)

    stochastic = rng.lognormal(mean=-0.5 * 0.15**2, sigma=0.15)

    return season_price * stochastic

def simulate_wind_farm_OandM(
    n_turbines: int,
    capacity_per_turbine: float,
    horizon_years: float = 25.0,         # base price in kNOK/MWh (e.g., 0.5 = 500 NOK/MWh)
    discount_rate: float = 0.07,
    cfd_active: bool = False,
    cfd_strike_price: float = 1.15,      # CfD strike price in kNOK/MWh (e.g., 1.15 = 1150 NOK/MWh)
    cfd_duration_years: float = 15.0,
    labor_rate: float = 0.5,             # labor cost in kNOK per technician-hour (e.g., 0.5 = 500 NOK/hr)
    hlv_mobilization: float = 5000.0,    # heavy-lift vessel mobilization cost in kNOK
    hlv_daily_rate: float = 1000.0,      # heavy-lift vessel daily charter cost in kNOK/day
    baseline_om_per_mw_year: float = 100.0,  # fixed O&M cost in kNOK per MW-year (baseline)
    site: WindRegion = None,           # Added parameter: optional site configuration
    n_simulations: int = 10000,
    random_seed: int = None
):
    """
    Monte Carlo O&M cost simulation for offshore wind (outputs in kNOK).
    If a WindRegion object is provided, override relevant parameters accordingly.
    """
    rng = np.random.default_rng(seed=random_seed)
    # Override parameters with site attributes if provided
    if site is not None:
        total_cap = n_turbines * capacity_per_turbine
        capacity_factor = site.capacity_factor
        # Adjust baseline fixed O&M for distance & maturity (e.g., far distance or FOAK -> upper range)
        if site.distance_km > 100 or site.maturity.upper() == "FOAK":
            baseline_om_per_mw_year = 100.0
        elif site.distance_km < 50 and site.foundation_type.lower() == "fixed":
            baseline_om_per_mw_year = 50.0
        else:
            baseline_om_per_mw_year = 80.0
        # Adjust heavy-vessel costs for floating (assume easier major repairs via towing)
        if site.foundation_type.lower() == "floating":
            hlv_mobilization *= 0.5  # reduce jack-up mobilization cost
        hvdc_on = site.hvdc_connection
    else:
        total_cap = n_turbines * capacity_per_turbine
        hvdc_on = False

    # Component reliability & cost setup (kNOK, FOAK 15MW parameters)
    component_types = [
        ComponentType("Blades",    shape=0.75, scale=8.0,  is_major=True,  part_cost=2250.0, avg_repair_days=14.0, crew_count=6, crew_days=12.0, use_heavy_vessel=True),
        ComponentType("Gearbox",   shape=1.30, scale=15.0, is_major=True,  part_cost=5000.0, avg_repair_days=14.0, crew_count=8, crew_days=32.0, use_heavy_vessel=True),
        ComponentType("Generator", shape=1.50, scale=18.0, is_major=True,  part_cost=3000.0, avg_repair_days=12.0, crew_count=8, crew_days=24.0, use_heavy_vessel=True),
        ComponentType("ElecSystem",shape=1.00, scale=2.0,  is_major=False, part_cost=50.0,   avg_repair_days=1.0, crew_count=2, crew_days=2.0, use_heavy_vessel=False)
    ]
    # Annual fixed O&M cost for entire capacity (kNOK/year)
    baseline_total_year = baseline_om_per_mw_year * total_cap

    total_costs = np.zeros(n_simulations)
    target_years = list(range(5, int(np.ceil(horizon_years)) + 1, 5))
    if target_years[-1] < horizon_years:
        target_years.append(int(np.ceil(horizon_years)))
    cum_costs_by_year = {Y: np.zeros(n_simulations) for Y in target_years}

    for sim in range(n_simulations):
        events_pv = []
        # Simulate failures for each turbine in each park
        for _ in range(int(n_turbines)):
            for comp in component_types:
                failure_times = sample_failure_times(comp, horizon_years, rng)#, offshore_factor)
                for t_fail in failure_times:
                    # Compute cost of this failure event (kNOK)
                    event_cost = compute_event_cost(
                        comp, site, t_fail, rng,
                        capacity_per_turbine, capacity_factor,
                        labor_rate, hlv_daily_rate, hlv_mobilization,
                        cfd_active, cfd_strike_price, cfd_duration_years
                    )
                    # Discount to present value
                    pv_factor = (1 + discount_rate) ** -t_fail
                    events_pv.append((t_fail, pv_factor * event_cost))
        # Simulate rare HVDC outage, if applicable
        if hvdc_on:
            hvdc_failure = simulate_hvdc_failure(
                site, rng, horizon_years, total_cap, capacity_factor,
                discount_rate, lambda t: get_power_price(
                                                        t, rng,
                                                        cfd_active,
                                                        cfd_strike_price,
                                                        cfd_duration_years
                                                    )
            )
            events_pv.extend(hvdc_failure)

        # Add baseline O&M for each year (in PV terms)
        for year in range(1, int(np.ceil(horizon_years)) + 1):
            pv_factor = (1 + discount_rate) ** -year
            events_pv.append((year, pv_factor * baseline_total_year))
        # Sort events by time and accumulate cost
        events_pv.sort(key=lambda x: x[0])
        cum_pv = 0.0
        idx = 0
        for time, cost in events_pv:
            while idx < len(target_years) and time > target_years[idx]:
                year_key = target_years[idx]
                cum_costs_by_year[year_key][sim] = cum_pv
                idx += 1
            cum_pv += cost
        while idx < len(target_years):
            year_key = target_years[idx]
            cum_costs_by_year[year_key][sim] = cum_pv
            idx += 1
        total_costs[sim] = cum_pv

    # Compute distribution stats
    def summary(arr):
        p_zero_cost = (arr == 0).mean()  # Fraction of runs with cost == 0

        # Central tendency
        mean_val = float(np.mean(arr))
        p50_val = float(np.quantile(arr, 0.50))

        # Percentiles (best → worst)
        p5_val  = float(np.quantile(arr, 0.05))   # best 5%
        p25_val = float(np.quantile(arr, 0.25))
        p75_val = float(np.quantile(arr, 0.75))
        p95_val = float(np.quantile(arr, 0.95))   # worst 5%

        # Tail means
        lower_tail = arr[arr <= p5_val]    # best-case tail
        upper_tail = arr[arr >= p95_val]   # worst-case tail

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

    #lifetime_stats = summary(total_costs)
    #cum_stats = {Y: summary(cum_costs_by_year[Y]) for Y in target_years}
    marg_stats = {}
    prev = 0

    for Y in target_years:
        label = f"{prev}-{Y-1}"

        period_costs = cum_costs_by_year[Y] - (
            cum_costs_by_year[prev] if prev > 0 else 0.0
        )

        marg_stats[label] = period_costs

        prev = Y


    marg_stats = {period: summary(marg_stats[period]) for period in marg_stats}

    return {
        "n_turbines": n_turbines,
        "capacity_per_turbine": capacity_per_turbine,
        "total_capacity_mw": total_cap,
        # "mean": lifetime_stats["mean"], "p50": lifetime_stats["p50"],
        # "var95": lifetime_stats["var95"], "cvar95": lifetime_stats["cvar95"],
        # "target_years": target_years,
        # "cumulative_stats": cum_stats,
        "marginal_stats": marg_stats
    }



def pretty_print_results(results, currency="kNOK", cfd_active=False):
    """
    Nicely print summary statistics from the O&M Monte Carlo simulation.

    Parameters
    ----------
    results : dict
        Output dictionary from simulate_wind_farm_OandM()
    currency : str
        Currency label for costs (e.g., 'NOK', 'EUR')
    cfd_active : bool
        Whether CFD is active
    """
    def fmt(x):
        return f"{x:,.0f}"

    print("\n" + "=" * 60)
    print(f" Offshore Wind O&M Cost Risk Summary {'with subsidy' if cfd_active else 'without subsidy'}")
    print("=" * 60)

    # print(f"Number of wind parks:          {fmt(results['number_of_parks'])}")
    print(f"Number of turbines in park:    {fmt(results['n_turbines'])}")
    print(f"Capacity per turbine:          {fmt(results['capacity_per_turbine'])}")

    print("=" * 60)
    print("Marginal O&M cost per 5-year period")
    print("(discounted to today)")
    print("\n  Values per MW")
    print("=" * 60)

    for period, stats in results["marginal_stats"].items():
        start, end = map(int, period.split("-"))
        display_start = 2030 + start

        
        if display_start == 2045:
                continue


        print(f"Year {display_start}")
        print("-" * 60)      
        print(f"  CVaR 5% (best):   {fmt(stats['cvar5_low'] / results['total_capacity_mw'])} {currency}/MW")
        print(f"  P5 (best 5%):     {fmt(stats['p5'] / results['total_capacity_mw'])} {currency}/MW")
        print(f"  P25:              {fmt(stats['p25'] / results['total_capacity_mw'])} {currency}/MW")
        print(f"  P50 (median):     {fmt(stats['p50'] / results['total_capacity_mw'])} {currency}/MW")
        print(f"  P75:              {fmt(stats['p75'] / results['total_capacity_mw'])} {currency}/MW")
        print(f"  P95 (worst 5%):   {fmt(stats['p95'] / results['total_capacity_mw'])} {currency}/MW")
        print(f"  CVaR 95% (worst): {fmt(stats['cvar95_high'] / results['total_capacity_mw'])} {currency}/MW")
        print("=" * 60)

# ------------------------------------------------------------
# Example usage: Sørlige Nordsjø II (SN2)
# ------------------------------------------------------------

turbine_capacity_mw = 15
n_turbines = 20
total_capacity_mw = turbine_capacity_mw * n_turbines

# 1) Define the SN2 site
sn2_site = WindRegion(
    name="SN2",
    distance_km=152.0,            # far offshore
    foundation_type="fixed",      # bottom-fixed
    water_depth_m=60.0,
    hvdc_connection=True,         # HVDC export
    maturity="FOAK",              # conservative first-of-a-kind
    capacity_factor=0.55
)

# Area-level WindRegion objects:
Nordavind_site = WindRegion(
    name="Nordavind", distance_km=150.0, foundation_type="floating", water_depth_m=300.0,
    hvdc_connection=True, maturity="FOAK", capacity_factor=0.4935
)
Nordvest_site = WindRegion(
    name="Nordvest", distance_km=150.0, foundation_type="floating", water_depth_m=275.0,
    hvdc_connection=True, maturity="FOAK", capacity_factor=0.4830
)
Vestavind1_site = WindRegion(
    name="Vestavind1", distance_km=90.0, foundation_type="floating", water_depth_m=300.0,
    hvdc_connection=False, maturity="FOAK", capacity_factor=0.4890
)
Vestavind2_site = WindRegion(
    name="Vestavind2", distance_km=80.0, foundation_type="floating", water_depth_m=270.0,
    hvdc_connection=False, maturity="FOAK", capacity_factor=0.5120
)
Sorvest_site = WindRegion(
    name="Sorvest", distance_km=150.0, foundation_type="fixed", water_depth_m=70.0,
    hvdc_connection=True, maturity="FOAK", capacity_factor=0.5530
)

# Example individual site WindRegion:
NordavindA_site = WindRegion(
    name="NordavindA", distance_km=250.0, foundation_type="floating", water_depth_m=250.0,
    hvdc_connection=True, maturity="FOAK", capacity_factor=0.4935
)

sites = [sn2_site, Nordavind_site, Nordvest_site, Vestavind1_site,
         Vestavind2_site, Sorvest_site, NordavindA_site]  # ... plus any others


# 2) Run the O&M simulation (drop-in compatible)
# NOTE: n_turbines / capacity_per_turbine / number_of_parks
# are overridden internally when `site` is provided.
cfd = True  # toggle CfD on/off

result = simulate_wind_farm_OandM(
    n_turbines=n_turbines,                 
    capacity_per_turbine=turbine_capacity_mw,
    cfd_active=cfd,                     
    site=sn2_site,                # <-- SN2 configuration injected here
    n_simulations=5000,
    random_seed=123
)

# 3) Print results (unchanged downstream usage)
pretty_print_results(result, currency="kNOK", cfd_active=cfd)
