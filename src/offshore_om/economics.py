"""
Economic evaluation utilities for offshore wind O&M analysis.

This module provides functions for discounting maintenance costs, valuing
lost electricity production, annualizing lifetime costs, and calculating
risk statistics across Monte Carlo simulations.

It also contains data-processing utilities for combining TIMES production,
capacity, and market-price data to estimate the discounted CAPEX-equivalent
value of Contracts for Difference subsidies.
"""

from pathlib import Path
import re
import pandas as pd
import numpy as np
from collections import defaultdict

ZERO_TOLERANCE = 1e-2
HOURS_PER_YEAR = 8760

def pv_factor(time_years: float, discount_rate: float) -> float:
    """
    Calculate the present-value discount factor for a future cash flow.

    Parameters
    ----------
    time_years : float
        Time between the valuation date and the cash flow, in years.
    discount_rate : float
        Annual discount rate expressed as a decimal.

    Returns
    -------
    float
        Present-value factor for the specified time and discount rate.
    """

    return (1.0 + discount_rate) ** (-time_years)

def capital_recovery_factor(discount_rate: float, horizon_years: float) -> float:
    """
    Calculate the capital recovery factor for a finite horizon.

    The capital recovery factor converts a present value into an equivalent
    constant annual value over the specified horizon.

    Parameters
    ----------
    discount_rate : float
        Annual discount rate expressed as a decimal.
    horizon_years : float
        Economic evaluation horizon in years.

    Returns
    -------
    float
        Capital recovery factor.

    Notes
    -----
    When the discount rate is zero, the factor is calculated as the reciprocal
    of the evaluation horizon.
    """
    r = discount_rate
    n = horizon_years

    if r == 0:
        return 1.0 / n

    return r * (1.0 + r) ** n / ((1.0 + r) ** n - 1.0)

def weighted_market_price_for_event(
    event,
    region_name: str,
    market_price_dict: dict,
):
    """
    Calculate the time-weighted market price during a downtime event.

    The downtime interval is divided across seasonal boundaries. The market
    price associated with each interval is weighted by the number of downtime
    hours occurring within that interval.

    Parameters
    ----------
    event : CostEvent
        Cost event containing downtime, lost production, and downtime timing
        information.
    region_name : str
        Electricity market region used to retrieve market prices.
    market_price_dict : dict
        Nested market-price data indexed by region, year, and season.

    Returns
    -------
    float
        Time-weighted market price in kNOK/MWh. Returns zero when the event
        has no positive downtime or lost production.

    Raises
    ------
    ValueError
        If a positive downtime event does not contain both
        ``downtime_start_years`` and ``downtime_end_years``.

    Notes
    -----
    Seasonal boundaries occur at fractions 0.25, 0.50, 0.75, and 1.00 of
    each simulation year.
    """
    if event.downtime_hours <= 0 or event.lost_mwh <= 0:
        return 0.0

    if event.downtime_start_years is None or event.downtime_end_years is None:
        raise ValueError("Event must contain downtime_start_years and downtime_end_years.")

    start_h = event.downtime_start_years * HOURS_PER_YEAR
    end_h = event.downtime_end_years * HOURS_PER_YEAR
    downtime_h = end_h - start_h

    weighted_price = 0.0
    current_h = start_h

    while current_h < end_h:
        t_years = current_h / HOURS_PER_YEAR

        price = get_market_price(
            t=t_years,
            region_name=region_name,
            market_price_dict=market_price_dict,
        )

        year_fraction = t_years % 1.0

        if year_fraction < 0.25:
            next_fraction = 0.25
        elif year_fraction < 0.50:
            next_fraction = 0.50
        elif year_fraction < 0.75:
            next_fraction = 0.75
        else:
            next_fraction = 1.00

        current_year = int(t_years)
        boundary_h = current_year * HOURS_PER_YEAR + next_fraction * HOURS_PER_YEAR
        interval_end_h = min(boundary_h, end_h)

        hours_in_interval = interval_end_h - current_h
        weighted_price += price * hours_in_interval

        current_h = interval_end_h

    return weighted_price / downtime_h

def evaluate_events_economics(
    events,
    discount_rate: float,
    horizon_years: float,
    region_name: str,
    market_scenarios: dict[str, dict] | None = None,
    subsidy_prices_ore_per_kwh: list[float] | None = None,
):
    """
    Evaluate the discounted economic consequences of a sequence of events.

    Direct O&M costs are discounted according to the occurrence time of each
    event. Lost electricity production can be valued using one or more market
    price scenarios, one or more subsidy prices, or both.

    The resulting present values are converted into equivalent annual costs
    using the capital recovery factor.

    Parameters
    ----------
    events : iterable of CostEvent
        Simulation events to evaluate.
    discount_rate : float
        Annual discount rate expressed as a decimal.
    horizon_years : float
        Economic evaluation horizon in years.
    region_name : str
        Electricity market region used when retrieving market prices.
    market_scenarios : dict of str to dict, optional
        Market-price scenarios. Keys identify the scenarios and values contain
        nested market-price data indexed by region, year, and season.
    subsidy_prices_ore_per_kwh : list of float, optional
        Subsidy prices in ore per kWh used as alternative valuations of lost
        electricity production.

    Returns
    -------
    dict
        Economic evaluation containing:

        ``"discount_rate"``
            Discount rate used in the evaluation.

        ``"CRF"``
            Capital recovery factor for the evaluation horizon.

        ``"direct_pv"``
            Present value of direct O&M costs.

        ``"lost_prod_pv_by_valuation"``
            Present value of lost production by valuation scenario.

        ``"total_pv_by_valuation"``
            Combined direct-cost and lost-production present value by
            valuation scenario.

        ``"annualized_by_valuation"``
            Equivalent annual total cost by valuation scenario.

    Notes
    -----
    Direct costs are discounted using the event occurrence time. Lost
    production is valued only for events with positive lost MWh.
    """

    crf = capital_recovery_factor(discount_rate, horizon_years)

    direct_pv = 0.0

    lost_prod_pv_by_valuation = {}

    for event in events:
        factor = pv_factor(event.time_years, discount_rate)

        direct_pv += event.direct_cost * factor

        if market_scenarios is not None and event.lost_mwh > 0:
            for market_label, market_price_dict in market_scenarios.items():
                price_knok_per_mwh = weighted_market_price_for_event(
                    event=event,
                    region_name=region_name,
                    market_price_dict=market_price_dict,
                )

                lost_value = event.lost_mwh * price_knok_per_mwh
                lost_value_pv = lost_value * factor

                valuation_name = f"{market_label}"
                lost_prod_pv_by_valuation[valuation_name] = (
                    lost_prod_pv_by_valuation.get(valuation_name, 0.0)
                    + lost_value_pv
                )

        if subsidy_prices_ore_per_kwh is not None and event.lost_mwh > 0:
            for subsidy_price in subsidy_prices_ore_per_kwh:
                price_knok_per_mwh = convert_ore_kwh_to_knok_mwh(subsidy_price)

                lost_value = event.lost_mwh * price_knok_per_mwh
                lost_value_pv = lost_value * factor

                valuation_name = f"Subsidy_{subsidy_price}"
                lost_prod_pv_by_valuation[valuation_name] = (
                    lost_prod_pv_by_valuation.get(valuation_name, 0.0)
                    + lost_value_pv
                )

    total_pv_by_valuation = {
        valuation_name: direct_pv + lost_prod_pv
        for valuation_name, lost_prod_pv in lost_prod_pv_by_valuation.items()
    }

    annualized_by_valuation = {
        valuation_name: total_pv * crf
        for valuation_name, total_pv in total_pv_by_valuation.items()
    }

    return {
        "discount_rate": discount_rate,
        "CRF": crf,
        "direct_pv": direct_pv,
        "lost_prod_pv_by_valuation": lost_prod_pv_by_valuation,
        "total_pv_by_valuation": total_pv_by_valuation,
        "annualized_by_valuation": annualized_by_valuation,
    }

def evaluate_monte_carlo_economics(
    mc_result: dict,
    discount_rates: list[float],
    horizon_years: float,
    region_name: str,
    market_scenarios: dict[str, dict] | None = None,
    subsidy_prices_ore_per_kwh: list[float] | None = None,
):
    """
    Evaluate annualized economic outcomes across Monte Carlo simulations.

    Each simulation realization is evaluated for the supplied discount rates
    and lost-production valuation scenarios. Annualized direct costs, lost
    production costs, and total costs are summarized using the mean, P75,
    P95, and CVaR95.

    Parameters
    ----------
    mc_result : dict
        Monte Carlo simulation results containing ``"events_by_simulation"``.
    discount_rates : list of float
        Annual discount rates expressed as decimals.
    horizon_years : float
        Economic evaluation horizon in years.
    region_name : str
        Electricity market region used for market-price valuation.
    market_scenarios : dict of str to dict, optional
        Market-price scenarios. Keys identify the scenarios and values contain
        nested market-price data indexed by region, year, and season.
    subsidy_prices_ore_per_kwh : list of float, optional
        Subsidy prices in ore per kWh used as alternative valuations of lost
        electricity production.

    Returns
    -------
    list of dict
        Economic result rows for each discount rate, valuation scenario, and
        statistical metric. Each row contains the direct cost, lost-production
        cost, total cost, capital recovery factor, and relevant scenario
        metadata.

    Notes
    -----
    Subsidy valuations are identified using the ``"Subsidy_"`` prefix. The
    corresponding subsidy price is included separately in the returned row.
    """
    events_by_simulation = mc_result["events_by_simulation"]

    results = []
    annualized_direct = []
    annualized_lost = defaultdict(list)
    annualized_total = defaultdict(list)

    for discount_rate in discount_rates:
        print(f"Runnig discount rate {discount_rate}")
        for sim_id, events in enumerate(events_by_simulation):
            econ = evaluate_events_economics(
                events=events,
                discount_rate=discount_rate,
                horizon_years=horizon_years,
                region_name=region_name,
                market_scenarios=market_scenarios,
                subsidy_prices_ore_per_kwh=subsidy_prices_ore_per_kwh,
            )

            annualized_direct.append(
                econ["direct_pv"] * econ["CRF"]
            )

            for valuation_name in econ["annualized_by_valuation"]:

                annualized_lost[valuation_name].append(
                    econ["lost_prod_pv_by_valuation"][valuation_name]
                    * econ["CRF"]
                )

                annualized_total[valuation_name].append(
                    econ["annualized_by_valuation"][valuation_name]
                )

        direct_stats = summary_stats(
            annualized_direct
        )

        for valuation in annualized_total:
            lost_stats = summary_stats(
                annualized_lost[valuation]
            )

            total_stats = summary_stats(
                annualized_total[valuation]
            )

            if valuation.startswith("Subsidy_"):

                subsidy_price = float(
                    valuation.replace("Subsidy_", "")
                )

            else:

                subsidy_price = None

            for metric in ["Mean", "P75", "P95", "CVaR95"]:

                results.append(
                    {
                        "Metric": metric,

                        "DiscountRate": discount_rate,

                        "CRF": econ["CRF"],

                        "Scenario": valuation,

                        "SubsidyPrice": subsidy_price,

                        "DirectCost": direct_stats[metric],

                        "LostCost": lost_stats[metric],

                        "TotalCost": total_stats[metric],
                    }
                )

    return results

def summary_stats(values):
    """
    Calculate central and upper-tail statistics.

    Parameters
    ----------
    values : array-like
        Numerical values to summarize.

    Returns
    -------
    dict
        Dictionary containing the mean, 75th percentile, 95th percentile,
        and CVaR95.

    Notes
    -----
    CVaR95 is calculated as the mean of values greater than or equal to the
    empirical 95th-percentile threshold.
    """

    values = np.asarray(values)

    mean_val = float(np.mean(values))
    p75_val = float(np.quantile(values, 0.75))
    p95_val = float(np.quantile(values, 0.95))

    upper_tail = values[values >= p95_val]

    cvar95_high = (
        float(np.mean(upper_tail))
        if upper_tail.size > 0
        else p95_val
    )

    return {
        "Mean": mean_val,
        "P75": p75_val,
        "P95": p95_val,
        "CVaR95": cvar95_high,
    }

def read_market_price_file(filepath):
    """
    Read seasonal electricity market prices from a CSV file.

    The input table is converted into a nested dictionary indexed by market
    region, model period, and season.

    Parameters
    ----------
    filepath : path-like
        Path to the market-price CSV file.

    Returns
    -------
    dict
        Nested market-price dictionary with the structure
        ``market_price[region][period][season]``.

    Notes
    -----
    The input file must contain the columns ``"Season"``, ``"Region"``,
    ``"Average PV (ore/kWh)"``, and ``"Period"``.
    """

    df = pd.read_csv(filepath)

    market_price = {}
    for _, row in df.iterrows():
        season = row["Season"]
        region = row["Region"]
        price = row["Average PV (ore/kWh)"]
        period = str(row["Period"])

        market_price.setdefault(region, {})
        market_price[region].setdefault(period, {})
        market_price[region][period][season] = price

    return market_price


def convert_ore_kwh_to_knok_mwh(price_ore_kwh):
    """
    Convert an electricity price from ore/kWh to kNOK/MWh.

    Parameters
    ----------
    price_ore_kwh : float
        Electricity price in ore per kWh.

    Returns
    -------
    float
        Electricity price in kNOK per MWh.
    """
    return price_ore_kwh * 0.01


def get_season_from_time(t):
    """
    Map simulation time to a season.

    Each simulation year is divided into four equal seasonal intervals.

    Parameters
    ----------
    t : float
        Simulation time in years.

    Returns
    -------
    str
        ``"Spring"``, ``"Summer"``, ``"Fall"``, or ``"Winter"``.
    """
    decimal_part = t % 1.0
    if decimal_part < 0.25:
        return "Spring"
    elif decimal_part < 0.50:
        return "Summer"
    elif decimal_part < 0.75:
        return "Fall"
    return "Winter"


def get_market_year_from_time(t):
    """
    Map simulation time to a market-price model year.

    Market years are assigned in five-year increments beginning in 2030 and
    are capped at 2050.

    Parameters
    ----------
    t : float
        Simulation time in years.

    Returns
    -------
    str
        Market-price model year.
    """
    integer_part = int(t)
    year_offset = (integer_part // 5) * 5
    market_year = 2030 + year_offset
    return str(min(market_year, 2050))


def get_market_price(t, region_name, market_price_dict):
    """
    Retrieve the market price for a simulation time and region.

    The simulation time is converted into a season and market-price model year
    before the corresponding value is retrieved and converted to kNOK/MWh.

    Parameters
    ----------
    t : float
        Simulation time in years.
    region_name : str
        Electricity market region.
    market_price_dict : dict
        Nested market-price data indexed by region, year, and season.

    Returns
    -------
    float or None
        Market price in kNOK/MWh, or ``None`` if the required region, year,
        or season is unavailable.
    """
    season = get_season_from_time(t)
    year = get_market_year_from_time(t)

    try:
        price_ore_kwh = market_price_dict[region_name][year][season]
        return convert_ore_kwh_to_knok_mwh(price_ore_kwh)
    except KeyError:
        return None


def get_market_region_name(region_name):
    """
    Map an offshore wind region to its Norwegian electricity price region.

    Parameters
    ----------
    region_name : str
        Offshore wind region name.

    Returns
    -------
    str
        Corresponding electricity price region. If no explicit mapping is
        defined, the original region name is returned.
    """
    if region_name.startswith("Sorvest") or region_name == "Sonnavind" or region_name == "Vestavind2":
        return "NO2"
    if region_name == "Nordvest":
        return "NO3"
    if region_name == "Nordavind":
        return "NO4"
    if region_name == "Vestavind1":
        return "NO5"
    return region_name


def ore_kwh_to_knok_gwh(price_ore_kwh: float) -> float:
    """
    Convert an electricity price from ore/kWh to kNOK/GWh.

    Parameters
    ----------
    price_ore_kwh : float
        Electricity price in ore per kWh.

    Returns
    -------
    float
        Electricity price in kNOK per GWh.

    Notes
    -----
    The conversion uses 1 ore equal to 0.01 NOK, 1 GWh equal to 1,000,000
    kWh, and 1 kNOK equal to 1,000 NOK.
    """
    return price_ore_kwh * 10.0


def read_market_price_table(filepath):
    """
    Read and prepare a TIMES market-price table.

    Market prices are converted from ore/kWh to kNOK/GWh, and model periods
    are converted to strings for subsequent table merges.

    Parameters
    ----------
    filepath : path-like
        Path to the market-price CSV file.

    Returns
    -------
    pandas.DataFrame
        Market-price table containing ``"TimeSliceSorted"``, ``"Region"``,
        ``"Period"``, ``"MarketPrice_ore_per_kWh"``, and
        ``"MarketPrice_kNOK_per_GWh"``.

    Raises
    ------
    ValueError
        If the input file is missing one or more required columns.

    Notes
    -----
    The required input columns are ``"TimeSliceSorted"``, ``"Region"``,
    ``"Average PV (ore/kWh)"``, and ``"Period"``.
    """
    df = pd.read_csv(filepath)

    required_cols = {
        "TimeSliceSorted",
        "Region",
        "Average PV (ore/kWh)",
        "Period",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Market price file is missing columns: {missing}")

    df["Period"] = df["Period"].astype(str)
    df["MarketPrice_ore_per_kWh"] = pd.to_numeric(
        df["Average PV (ore/kWh)"], errors="coerce"
    )

    df["MarketPrice_kNOK_per_GWh"] = df["MarketPrice_ore_per_kWh"].apply(
        ore_kwh_to_knok_gwh
    )

    return df[
        [
            "TimeSliceSorted",
            "Region",
            "Period",
            "MarketPrice_ore_per_kWh",
            "MarketPrice_kNOK_per_GWh",
        ]
    ].copy()

def read_capacity_file(filepath):
    """
    Read and prepare a TIMES installed-capacity table.

    Parameters
    ----------
    filepath : path-like
        Path to the installed-capacity CSV file.

    Returns
    -------
    pandas.DataFrame
        Capacity table containing ``"Period"``, ``"Process"``, ``"Region"``,
        and ``"InstalledCapacity_MW"``.

    Raises
    ------
    ValueError
        If the input file is missing one or more required columns.

    Notes
    -----
    The required input columns are ``"Period"``, ``"Process"``,
    ``"Sum of PV"``, and ``"Region"``. Capacity values with an absolute
    magnitude below ``ZERO_TOLERANCE`` are set to zero.
    """

    df = pd.read_csv(filepath)

    required_cols = {
        "Period",
        "Process",
        "Sum of PV",
        "Region",
    }

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"Capacity file is missing columns: {missing}"
        )

    df["Period"] = df["Period"].astype(str)

    df["InstalledCapacity_MW"] = pd.to_numeric(
        df["Sum of PV"],
        errors="coerce"
    ).fillna(0.0)

    df["InstalledCapacity_MW"] = (
        df["InstalledCapacity_MW"]
        .where(df["InstalledCapacity_MW"].abs() >= ZERO_TOLERANCE, 0.0)
        )

    return df[
        [
            "Period",
            "Process",
            "Region",
            "InstalledCapacity_MW",
        ]
    ].copy()

def extract_period_from_filename(filepath: Path) -> str:
    """
    Extract a four-digit model year from a filename.

    Parameters
    ----------
    filepath : pathlib.Path
        Path whose filename contains a year preceded by a hyphen.

    Returns
    -------
    str or None
        Extracted year between 2000 and 2099, or ``None`` if no matching year
        is found.

    Examples
    --------
    ``production-2030.csv`` returns ``"2030"``.

    ``SorvestF-2035.csv`` returns ``"2035"``.

    ``results-2040-test.csv`` returns ``"2040"``.
    """
    match = re.search(r"-(20\d{2})", filepath.stem)

    if match is None:
        return None

    return match.group(1)

def read_production_files(
    production_folder,
    file_pattern="*.csv",
    production_col="Sum of PV",
    ):

    """
    Read and combine TIMES production files.

    Files matching the specified pattern are read from a directory. The model
    period is extracted from each filename, and the selected production column
    is standardized as ``"Production_GWh"``.

    Parameters
    ----------
    production_folder : path-like
        Directory containing the production CSV files.
    file_pattern : str, optional
        Glob pattern used to select files. The default is ``"*.csv"``.
    production_col : str, optional
        Name of the input column containing production values. The default is
        ``"Sum of PV"``.

    Returns
    -------
    pandas.DataFrame
        Combined production records with standardized period and production
        columns.

    Raises
    ------
    FileNotFoundError
        If no files match ``file_pattern`` in ``production_folder``.
    ValueError
        If a selected file is missing ``"TimeSliceSorted"``, ``"Process"``,
        ``"Region"``, or the specified production column.

    Notes
    -----
    Files without a model year matching the expected filename pattern are
    skipped.
    """

    production_folder = Path(production_folder)

    files = sorted(production_folder.glob(file_pattern))

    if not files:
        raise FileNotFoundError(
        f"No production files found in {production_folder} with pattern {file_pattern}"
        )

    dfs = []

    for file in files:
        period = extract_period_from_filename(file)

        if period is None:
            continue

        df = pd.read_csv(file)
        df["Period"] = str(period)
        required_cols = {"TimeSliceSorted", production_col, "Process", "Region"}
        missing = required_cols - set(df.columns)

        if missing:
            raise ValueError(f"{file.name} is missing columns: {missing}")

        df = df.rename(columns={production_col: "Production_GWh"})

        dfs.append(df)

    production = pd.concat(dfs, ignore_index=True)

    production["Period"] = production["Period"].astype(str)

    production["Production_GWh"] = pd.to_numeric(

    production["Production_GWh"], errors="coerce"
    ).fillna(0.0)

    return production

def make_subsidy_price_table(subsidy_price_ore_kwh, periods, regions=None):
    """
    Construct a subsidy-price table by region and model period.

    Subsidy prices may be provided as one scalar value, as values indexed by
    period, or as values indexed by region-period pairs.

    Parameters
    ----------
    subsidy_price_ore_kwh : float or dict
        Subsidy-price specification in ore/kWh. Supported forms are:

        * One scalar applied to all requested regions and periods.
        * A dictionary indexed by period.
        * A dictionary indexed by ``(region, period)`` tuples.
    periods : iterable
        Model periods included in the output table.
    regions : iterable, optional
        Regions included in the output table. If omitted, a region value of
        ``None`` is used.

    Returns
    -------
    pandas.DataFrame
        Subsidy-price table containing ``"Region"``, ``"Period"``,
        ``"SubsidyPrice_ore_per_kWh"``, and
        ``"SubsidyPrice_kNOK_per_GWh"``.

    Raises
    ------
    TypeError
        If ``subsidy_price_ore_kwh`` is neither a scalar nor a dictionary.
    ValueError
        If the supplied data produces an empty subsidy-price table.

    Notes
    -----
    Dictionary lookup prioritizes a matching ``(region, period)`` entry,
    followed by a string period and then an integer period.
    """
    periods = [str(p) for p in periods]

    if regions is None:
        regions = [None]
    else:
        regions = list(regions)

    rows = []

    if isinstance(subsidy_price_ore_kwh, (int, float)):
        for region in regions:
            for period in periods:
                rows.append(
                    {
                        "Region": region,
                        "Period": period,
                        "SubsidyPrice_ore_per_kWh": float(subsidy_price_ore_kwh),
                    }
                )

    elif isinstance(subsidy_price_ore_kwh, dict):
        for region in regions:
            for period in periods:
                value = None

                if (region, period) in subsidy_price_ore_kwh:
                    value = subsidy_price_ore_kwh[(region, period)]
                elif period in subsidy_price_ore_kwh:
                    value = subsidy_price_ore_kwh[period]
                elif int(period) in subsidy_price_ore_kwh:
                    value = subsidy_price_ore_kwh[int(period)]

                if value is not None:
                    rows.append(
                        {
                            "Region": region,
                            "Period": period,
                            "SubsidyPrice_ore_per_kWh": float(value),
                        }
                    )

    else:
        raise TypeError(
            "subsidy_price_ore_kwh must be a scalar or a dictionary."
        )

    subsidy = pd.DataFrame(rows)

    if subsidy.empty:
        raise ValueError("Subsidy price table is empty. Check subsidy input.")

    subsidy["SubsidyPrice_kNOK_per_GWh"] = subsidy[
        "SubsidyPrice_ore_per_kWh"
    ].apply(ore_kwh_to_knok_gwh)

    return subsidy


def get_wind_region(region, process):
    """
    Map a TIMES region and process to an offshore wind region.

    Processes in the ``"O_Sorvest"`` region are mapped according to their
    process suffix. For other regions, the ``"O_"`` prefix is removed.

    Parameters
    ----------
    region : str
        TIMES region identifier.
    process : str
        TIMES process identifier.

    Returns
    -------
    str
        Offshore wind region name.
    """

    if region == "O_Sorvest":

        if process.endswith("_S1"):
            return "SorvestA"
        elif process.endswith("_S2"):
            return "SorvestB"
        elif process.endswith("_S3"):
            return "SorvestC"
        elif process.endswith("_S4"):
            return "SorvestD"
        elif process.endswith("_S5"):
            return "SorvestE"
        elif process.endswith("_S6"):
            return "SorvestF"
        elif process.endswith("_S7"):
            return "Sonnavind"

    return region.replace("O_", "")


def calculate_cfd_subsidy_capex(
    production_folder,
    market_price_file,
    capacity_file,
    subsidy_price_ore_kwh,
    region_price_map,
    discount_rate=0.04,
    base_year=2030,
    file_pattern="*.csv",
    process_filter=None,
    output_folder=None,
):
    """
    Calculate the discounted CAPEX-equivalent value of CfD payments.

    Production data are combined with seasonal market prices, installed
    capacity, and subsidy prices. The resulting Contract for Difference
    payments are normalized by installed capacity, annualized over each
    five-year model period, discounted to the base year, and aggregated by
    offshore wind region.

    Parameters
    ----------
    production_folder : path-like
        Directory containing TIMES production CSV files.
    market_price_file : path-like
        Path to the seasonal market-price CSV file.
    capacity_file : path-like
        Path to the installed-capacity CSV file.
    subsidy_price_ore_kwh : float or dict
        Subsidy-price specification in ore/kWh. It may be a scalar, a
        dictionary indexed by period, or a dictionary indexed by
        ``(region, period)``.
    region_price_map : dict
        Mapping from TIMES production regions to electricity price regions.
    discount_rate : float, optional
        Annual discount rate expressed as a decimal. The default is 0.04.
    base_year : int, optional
        Year to which CfD payments are discounted. The default is 2030.
    file_pattern : str, optional
        Glob pattern used to select production files. The default is
        ``"*.csv"``.
    process_filter : str or iterable of str, optional
        Process identifier or identifiers retained in the production data.
        If omitted, all processes are retained.
    output_folder : path-like, optional
        Directory in which ``cfd_capex.csv`` is exported. If omitted, no CSV
        file is written.

    Returns
    -------
    pandas.DataFrame
        Discounted CAPEX-equivalent CfD support by wind region. The returned
        table contains ``"WindRegion"`` and
        ``"Discounted_CAPEX_kNOK_per_MW"``.

    Raises
    ------
    FileNotFoundError
        If no production files match the selected file pattern.
    ValueError
        If required columns are missing from an input file.
    ValueError
        If a production region has no corresponding market-price region.
    ValueError
        If market prices are missing for positive-production records.
    ValueError
        If subsidy prices are missing for positive-production records.

    Notes
    -----
    CfD payments are calculated as production multiplied by the difference
    between the subsidy price and the market price. Negative differences are
    retained by the calculation.

    Rows with installed capacity less than or equal to ``ZERO_TOLERANCE`` are
    excluded. Each model-period payment is treated as an annual value over a
    five-year period before discounting.
    """
    production = read_production_files(
        production_folder=production_folder,
        file_pattern=file_pattern,
    )
    production["Production_GWh"] = production["Production_GWh"].where(
        production["Production_GWh"].abs() >= ZERO_TOLERANCE,
        0.0,
    )

    if process_filter is not None:
        if isinstance(process_filter, str):
            process_filter = [process_filter]

        production = production[production["Process"].isin(process_filter)].copy()

    production["PriceRegion"] = production["Region"].map(region_price_map)

    missing_region_map = production.loc[
        production["PriceRegion"].isna(), "Region"
    ].drop_duplicates()

    if not missing_region_map.empty:
        raise ValueError(
            "Missing region mapping for production regions: "
            + ", ".join(missing_region_map.astype(str))
        )

    market_price = read_market_price_table(market_price_file)
    market_price = market_price.rename(columns={"Region": "PriceRegion"})

    merged = production.merge(
        market_price,
        on=["TimeSliceSorted", "PriceRegion", "Period"],
        how="left",
        validate="many_to_one",
    )

    capacity = read_capacity_file(capacity_file)
    
    merged = merged.merge(
        capacity,
        on=["Period", "Process", "Region"],
        how="left"
    )

    merged = merged.loc[
        merged["InstalledCapacity_MW"] > ZERO_TOLERANCE
    ].copy()

    missing_market = merged[
        (merged["Production_GWh"].abs() > 1e-12)
        & (merged["MarketPrice_kNOK_per_GWh"].isna())
    ]

    if not missing_market.empty:
        example = missing_market[
            ["TimeSliceSorted", "Region", "PriceRegion", "Period", "Production_GWh"]
        ].drop_duplicates().head(20)

        raise ValueError(
            "Market price is missing for some positive-production rows. "
            "Here are examples:\n"
            + example.to_string(index=False)
        )

    merged["MarketPrice_ore_per_kWh"] = merged["MarketPrice_ore_per_kWh"].fillna(0.0)
    merged["MarketPrice_kNOK_per_GWh"] = merged[
        "MarketPrice_kNOK_per_GWh"
    ].fillna(0.0)

    periods = sorted(merged["Period"].unique())
    regions = sorted(merged["Region"].unique())

    subsidy = make_subsidy_price_table(
        subsidy_price_ore_kwh=subsidy_price_ore_kwh,
        periods=periods,
        regions=regions,
    )

    merged = merged.merge(
        subsidy,
        on=["Region", "Period"],
        how="left",
        validate="many_to_one",
    )

    missing_subsidy = merged[
        (merged["Production_GWh"].abs() > 1e-12)
        & (merged["SubsidyPrice_kNOK_per_GWh"].isna())
    ]

    if not missing_subsidy.empty:
        example = missing_subsidy[
            ["TimeSliceSorted", "Region", "Period", "Production_GWh"]
        ].drop_duplicates().head(20)

        raise ValueError(
            "Subsidy price is missing for some positive-production rows. "
            "Here are examples:\n"
            + example.to_string(index=False)
        )

    merged["SubsidyPrice_ore_per_kWh"] = merged[
        "SubsidyPrice_ore_per_kWh"
    ].fillna(0.0)

    merged["SubsidyPrice_kNOK_per_GWh"] = merged[
        "SubsidyPrice_kNOK_per_GWh"
    ].fillna(0.0)

    merged["CfD_Delta_ore_per_kWh"] = (
        merged["SubsidyPrice_ore_per_kWh"] - merged["MarketPrice_ore_per_kWh"]
    )

    merged["CfD_Delta_kNOK_per_GWh"] = (
        merged["SubsidyPrice_kNOK_per_GWh"]
        - merged["MarketPrice_kNOK_per_GWh"]
    )

    merged["CfD_payment_kNOK"] = (
        merged["Production_GWh"] * merged["CfD_Delta_kNOK_per_GWh"]
    )

    merged["CfD_payment_kNOK_per_MW"] = np.where(
        merged["InstalledCapacity_MW"] > 0,
        merged["CfD_payment_kNOK"] / merged["InstalledCapacity_MW"],
        0.0
    )
    merged["WindRegion"] = merged.apply(
        lambda row: get_wind_region(
            row["Region"],
            row["Process"]
        ),
        axis=1
    )
    
    annuity_factor = (
        1 - (1 + discount_rate) ** (-5)
    ) / discount_rate
    
    annual = (
        merged.groupby(
            ["WindRegion", "Period"],
            as_index=False
        )
        .agg(
            Annual_CfD_kNOK_per_MW=(
                "CfD_payment_kNOK_per_MW",
                "sum"
            )
        )
    )

    annual["DiscountYears"] = (
        annual["Period"].astype(int)
        - int(base_year)
    )

    annual["DiscountFactor"] = (
        1 / (1 + discount_rate) ** annual["DiscountYears"]
    )

    annual["PV_CfD_kNOK_per_MW"] = (
        annual["Annual_CfD_kNOK_per_MW"]
        * annuity_factor
        * annual["DiscountFactor"]
    )

    capex = (
        annual.groupby(
            ["WindRegion"],
            as_index=False
        )
        .agg(
            Discounted_CAPEX_kNOK_per_MW=(
                "PV_CfD_kNOK_per_MW",
                "sum"
            )
        )
    )

    if output_folder is not None:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        capex.to_csv(output_folder / "cfd_capex.csv", index=False)

    return capex
