from pathlib import Path
import re
import pandas as pd
import numpy as np
from collections import defaultdict

ZERO_TOLERANCE = 1e-2
HOURS_PER_YEAR = 8760

def pv_factor(time_years: float, discount_rate: float) -> float:
    return (1.0 + discount_rate) ** (-time_years)

def capital_recovery_factor(discount_rate: float, horizon_years: float) -> float:
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
    events_by_simulation = mc_result["events_by_simulation"]

    results = []
    annualized_direct = []
    annualized_lost = defaultdict(list)
    annualized_total = defaultdict(list)

    for discount_rate in discount_rates:
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

                    "Scenario": valuation,

                    "SubsidyPrice": subsidy_price,

                    "DirectCost": direct_stats[metric],

                    "LostCost": lost_stats[metric],

                    "TotalCost": total_stats[metric],
                }
            )

    return results

def summary_stats(values):

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
    return price_ore_kwh * 0.01


def get_season_from_time(t):
    decimal_part = t % 1.0
    if decimal_part < 0.25:
        return "Spring"
    elif decimal_part < 0.50:
        return "Summer"
    elif decimal_part < 0.75:
        return "Fall"
    return "Winter"


def get_market_year_from_time(t):
    integer_part = int(t)
    year_offset = (integer_part // 5) * 5
    market_year = 2030 + year_offset
    return str(min(market_year, 2050))


def get_market_price(t, region_name, market_price_dict):
    season = get_season_from_time(t)
    year = get_market_year_from_time(t)

    try:
        price_ore_kwh = market_price_dict[region_name][year][season]
        return convert_ore_kwh_to_knok_mwh(price_ore_kwh)
    except KeyError:
        return None


def get_market_region_name(region_name):
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
    Convert øre/kWh to kNOK/GWh.

    1 øre = 0.01 NOK
    1 GWh = 1,000,000 kWh
    1 kNOK = 1,000 NOK
    """
    return price_ore_kwh * 10.0


def read_market_price_table(filepath):
    """
    Reads market price file and returns a DataFrame with prices in kNOK/GWh.

    Expected columns: TimeSliceSorted, Region, Average PV (ore/kWh), Period
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
    Reads installed capacity file.

    Expected columns:
    Period, Process, Sum of PV, Region

    Returns:
        Period
        Process
        Region
        InstalledCapacity_MW
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
    Extracts the year from filenames containing '-20xx'.

    Examples:
        production-2030.csv -> "2030"
        SorvestF-2035.csv   -> "2035"
        results-2040-test.csv -> "2040"
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
    Reads all production files named with xxx_<Period>.csv.
    Expected columns:
    TimeSliceSorted, Sum of PV, Process, Region
    Adds:
    Period
    Production_GWh
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
    Creates a subsidy price table.

    Accepts either:

    1) Scalar:
       subsidy_price_ore_kwh = 100

    2) Dict by period:
       subsidy_price_ore_kwh = {
           "2030": 100,
           "2035": 95,
           "2040": 90,
       }

    3) Dict by (region, period):
       subsidy_price_ore_kwh = {
           ("O_Sorvest", "2030"): 100,
           ("O_Sorvest", "2035"): 95,
           ("O_Vestavind2", "2030"): 105,
       }

    Returns columns:
    Region, Period, SubsidyPrice_ore_per_kWh, SubsidyPrice_kNOK_per_GWh
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
    Convert Region + Process into a TIMES WindRegion.
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
