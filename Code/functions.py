import numpy as np 
import pandas as pd
import pylab as pl
import scipy as sp
import scipy.stats as sps
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def smooth_keep_ends_savgol(y, window=7, poly=2):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return y.copy()

    # window must be odd and <= n
    window = min(window, n if n % 2 == 1 else n-1)
    if window < 3:
        return y.copy()

    y_s = savgol_filter(y, window_length=window, polyorder=min(poly, window-1))
    y_s[0] = y[0]
    y_s[-1] = y[-1]
    return y_s


paper = {
    "Generator": (1.52, 18.72),
    "Gearbox":   (1.38, 15.02),
    "Blade":    (0.75, 86.80),
}

offshore_factor = 1.26

observed_data = {
    "Generator": np.array([0.008, 0.009, 0.027, 0.033, 0.028,
                           0.063, 0.078, 0.036, 0.040, 0.024]),
    "Gearbox":   np.array([0.005, 0.027, 0.036, 0.078, 0.100,
                           0.058, 0.040, 0.074, 0.050, 0.036]),
    "Blade":     np.array([0.035, 0.015, 0.021, 0.028, 0.034,
                           0.026, 0.013, 0.010, 0.016, 0.004]),
}


def F(t, k, lam):
    return 1 - np.exp(-(t / lam)**k)

    
def m(t,k,lam):
    return (t/lam)**k


def R_hat(t, k, lam):
    return F(t, k, lam) - F(t - 1, k, lam)

    
def delta_m(t, k, lam):
    return m(t,k,lam) - m(t-1,k,lam)

def p_ge1(t, k, lam):
    return 1 - np.exp(-delta_m(t,k,lam))


def fit_weibull(R_obs):

    t = np.arange(1, len(R_obs) + 1)

    def objective(params):
        k, lam = params

        if k <= 0 or lam <= 0:
            return 1e10

        pred = R_hat(t, k, lam)
        return np.sum((R_obs - pred)**2)

    res = minimize(
        objective,
        x0=[1.5, 20],
        bounds=[(0.01, 10), (1, 200)],
        method="L-BFGS-B"
    )

    return res.x


def multistart(obj):
    k0s  = [0.3,0.5,0.8,1.0,1.5,2.0,3.0,5.0]
    l0s  = [5,10,15,20,30,40,60,80,100,150,200]
    best = None
    for k0 in k0s:
        for l0 in l0s:
            res = minimize(obj, [k0,l0], method="L-BFGS-B",
                           bounds=[(1e-6, 50), (1e-6, 500)])
            if best is None or res.fun < best.fun:
                best = res
    return best


def fit_nhpp_per_year_prob(R_obs):
    def obj(x):
        ti = np.arange(1, 11, dtype=float)
        k, lam = x
        if k<=0 or lam<=0: return 1e18
        pred = p_ge1(ti,k,lam)
        return float(np.sum((R_obs - pred)**2))
    res = multistart(obj)
    return res.x, res.fun


def simulate_weibull_failures(shape, scale, horizon_years, rng):
    """
    Simulate failure times for a Weibull NHPP (power-law process)
    with mean value function m(t) = (t/scale)^shape.

    Uses time-rescaling:
      S_n = sum Exp(1)
      T_n = scale * (S_n)^(1/shape)
    """
    shape = float(shape)
    scale = float(scale)

    #Adjusted scale for offshore wind
    scale_adjusted = scale / (offshore_factor ** (1.0 / shape))

    times = []
    s = 0.0  # cumulative intensity

    while True:
        s += rng.exponential(1.0)          # Exp(1)
        t = scale_adjusted * (s ** (1.0 / shape))   # inverse of m(t)

        if t > horizon_years:
            break

        times.append(t)

    return np.array(times)

# def simulate_renewal_with_downtime(shape, scale, horizon_years, downtime_hours, rng):
#     """
#     Perfect repair renewal process:
#     - Weibull time-to-failure
#     - after each failure: add downtime (no failures can occur during downtime)
#     - after downtime: component is 'as good as new' (clock resets)
#     Returns failure times in calendar years.
#     """
#     k = float(shape); lam = float(scale)
#     dt_years = float(downtime_hours) / 8760.0  # hours -> years

#     times = []
#     t = 0.0  # calendar time

#     while True:
#         # time to next failure while operating
#         ttf = lam * rng.weibull(k)
#         t += ttf
#         if t > horizon_years:
#             break

#         times.append(t)

#         # repair time: cannot fail during downtime
#         t += dt_years
#         if t > horizon_years:
#             break

#     return np.array(times)

def discounted_fault_cost(failure_time, direct_cost, installed_capacity_mw, utility_factor, power_price_per_mwh, downtime_hours, discount_rate):
    """
    Compute discounted cost of one failure.

    Parameters
    ----------
    failure_time : float
        Time of failure in years.
    direct_cost : float
        Direct repair cost.
    installed_capacity_mw : float
        Installed capacity in MW.
    utility_factor : float
        Capacity/utility factor, e.g. 0.24.
    power_price_per_mwh : float
        Electricity price in currency per MWh.
    downtime_hours : float
        Downtime in hours.
    discount_rate : float
        Annual discount rate, e.g. 0.08.

    Returns
    -------
    float
        Discounted fault cost.
    """
    # Opportunity cost = lost energy production * electricity price
    # Lost energy (MWh) = MW * utility_factor * downtime_hours
    opportunity_cost = (
        installed_capacity_mw * utility_factor * downtime_hours * power_price_per_mwh
    )

    undiscounted_cost = direct_cost + opportunity_cost
    discounted_cost = undiscounted_cost / ((1.0 + discount_rate) ** failure_time)

    return discounted_cost


def simulate_one_path(
    fault_data,
    horizon_years,
    installed_capacity_mw,
    utility_factor,
    power_price_per_mwh,
    discount_rate,
    rng,
):
    """
    Simulate one Monte Carlo path of lifecycle fault costs.

    Parameters
    ----------
    fault_data : list of dict
        Each dict describes one fault type:
        {
            "name": str,
            "shape": float,
            "scale": float,         # in years
            "direct_cost": float,
            "downtime_hours": float
        }

    horizon_years : float
        Simulation horizon.
    installed_capacity_mw : float
        Installed capacity.
    utility_factor : float
        Utility factor U.
    power_price_per_mwh : float
        Power price P.
    discount_rate : float
        Discount rate d.
    capex : float
        CAPEX used for normalization.
    rng : np.random.Generator
        RNG.

    Returns
    -------
    dict
        Path-level results and fault-level event records.
    """
    total_discounted_loss = 0.0
    records = []

    for fault in fault_data:
        fail_times = simulate_weibull_failures(
            shape=fault["shape"],
            scale=fault["scale"],
            horizon_years=horizon_years,
            rng=rng,
        )

        for t in fail_times:
            c = discounted_fault_cost(
                failure_time=max(1, int(np.floor(t))),
                direct_cost=fault["DC"],
                installed_capacity_mw=installed_capacity_mw,
                utility_factor=utility_factor,
                power_price_per_mwh=power_price_per_mwh,
                downtime_hours=fault["DT"],
                discount_rate=discount_rate,
            )

            total_discounted_loss += c

            records.append(
                {
                    "fault_type": fault["name"],
                    "time_years": t,
                    "direct_cost": fault["DC"],
                    "downtime_hours": fault["DT"],
                    "discounted_cost": c,
                }
            )

    if records:
        records_df = pd.DataFrame(records).sort_values("time_years").reset_index(drop=True)
    else:
        records_df = pd.DataFrame(
            columns=[
                "fault_type",
                "time_years",
                "direct_cost",
                "downtime_hours",
                "discounted_cost",
            ]
        )
        
    return {
        "lc": total_discounted_loss,
        "n_failures": len(records),
        "records": records_df,
    }

def simulate_one_farm_path(
    n_turbines,
    fault_data,
    horizon_years,
    installed_capacity_mw_per_turbine,
    utility_factor,
    power_price_per_mwh,
    discount_rate,
    rng,
):
    farm_lc = 0.0
    farm_records = []
    farm_failures = 0

    for turb_id in range(n_turbines):
        res = simulate_one_path(
            fault_data=fault_data,
            horizon_years=horizon_years,
            installed_capacity_mw=installed_capacity_mw_per_turbine,  # per turbine!
            utility_factor=utility_factor,
            power_price_per_mwh=power_price_per_mwh,
            discount_rate=discount_rate,
            rng=rng,
        )
        farm_lc += res["lc"]
        farm_failures += res["n_failures"]

        df = res["records"]
        if not df.empty:
            df = df.copy()
            df["turbine_id"] = turb_id
            farm_records.append(df)

    farm_records_df = (
        pd.concat(farm_records, ignore_index=True).sort_values("time_years").reset_index(drop=True)
        if farm_records else
        pd.DataFrame(columns=["fault_type","time_years","direct_cost","downtime_hours","discounted_cost","turbine_id"])
    )

    return {
        "lc": farm_lc,
        "n_failures": farm_failures,
        "records": farm_records_df,
    }

def monte_carlo_lifecycle_cost(
    n_sims,
    n_turbines,
    fault_data,
    horizon_years,
    installed_capacity_mw_per_turbine,
    utility_factor,
    power_price_per_mwh,
    discount_rate,
    capex_per_turbine,
    seed=42,
):
    rng = np.random.default_rng(seed)

    lc_values = np.zeros(n_sims)
    lc_pct_capex_values = np.zeros(n_sims)
    n_failures_values = np.zeros(n_sims)
    records_values = []

    capex_total = n_turbines * capex_per_turbine  # IMPORTANT

    for i in range(n_sims):
        res = simulate_one_farm_path(
            n_turbines=n_turbines,
            fault_data=fault_data,
            horizon_years=horizon_years,
            installed_capacity_mw_per_turbine=installed_capacity_mw_per_turbine,
            utility_factor=utility_factor,
            power_price_per_mwh=power_price_per_mwh,
            discount_rate=discount_rate,
            rng=rng,
        )

        lc_values[i] = res["lc"]
        lc_pct_capex_values[i] = 100 * res["lc"] / capex_total  # fraction; multiply by 100 if you want %
        n_failures_values[i] = res["n_failures"]

        df = res["records"]
        df["sim_id"] = i
        records_values.append(df)

    return {
        "lc_values": lc_values,
        "lc_pct_capex_values": lc_pct_capex_values,
        "n_failures_values": n_failures_values,
        "records_values": records_values,
        "capex_total": capex_total,
    }


def yearly_failure_percentage(records_values, n_sims, horizon_years):
    """
    Calculate percentage of runs with at least one failure
    for each fault type in each year.

    Parameters
    ----------
    records_values : list of pd.DataFrame
        One DataFrame per simulation run.
    n_sims : int
        Total number of Monte Carlo runs.
    horizon_years : int
        Number of years in simulation.

    Returns
    -------
    pd.DataFrame
        Columns:
        - year
        - fault_type
        - runs_with_failure
        - failure_pct
    """
    all_rows = []

    for sim_id, df in enumerate(records_values):
        if df.empty:
            continue

        temp = df.copy()

        # Convert time to simulation year
        temp["year"] = np.ceil(temp["time_years"]).astype(int)

        # Keep valid years only
        temp = temp[(temp["year"] >= 1) & (temp["year"] <= horizon_years)]

        temp["sim_id"] = sim_id

        all_rows.append(temp)

    if not all_rows:
        return pd.DataFrame(
            columns=["year", "fault_type", "runs_with_failure", "failure_pct"]
        )

    all_failures = pd.concat(all_rows, ignore_index=True)

    summary = (
        all_failures.groupby(["year", "fault_type"])["sim_id"]
        .nunique()
        .reset_index(name="runs_with_failure")
    )

    summary["failure_pct"] = summary["runs_with_failure"] / n_sims

    # Fill in missing year/fault combinations with zero
    all_fault_types = sorted(all_failures["fault_type"].unique())
    full_index = pd.MultiIndex.from_product(
        [range(1, horizon_years + 1), all_fault_types],
        names=["year", "fault_type"]
    )

    summary = (
        summary.set_index(["year", "fault_type"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    return summary


def var_cvar(losses, alpha=0.95):
    """
    Compute VaR and CVaR for a loss distribution.

    Parameters
    ----------
    losses : np.ndarray
        Loss samples.
    alpha : float
        Confidence level, e.g. 0.95.

    Returns
    -------
    tuple
        (VaR, CVaR)
    """
    var = np.quantile(losses, alpha)
    tail_losses = losses[losses >= var]
    cvar = tail_losses.mean() if len(tail_losses) > 0 else var
    return var, cvar

def plot_yearly_failure_percentage(summary_df, observed_data):
    """
    Create one subplot per fault type showing yearly failure percentage.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output from yearly_failure_percentage()
    """

    fault_types = sorted(summary_df["fault_type"].unique())
    n_faults = len(fault_types)

    fig, axes = plt.subplots(
        n_faults, 1,
        figsize=(10, 3 * n_faults),
        sharex=False
    )

    # If only one fault type, axes is not a list
    if n_faults == 1:
        axes = [axes]

    for ax, fault in zip(axes, fault_types):
        df = summary_df[summary_df["fault_type"] == fault].sort_values("year")
        y = df["failure_pct"]
        y_smooth = smooth_keep_ends_savgol(y)

        #ax.plot(df["year"], y, marker="o")
        ax.plot(df["year"], y_smooth, label="Rolling avg")
        
        if fault in observed_data:
            obs = observed_data[fault]
            years = np.arange(1, len(obs) + 1)
            ax.bar(years, obs, alpha=0.5, label="Observed (10y)")

        ax.set_title(fault)
        ax.set_ylabel("Probability")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, int(df["year"].max()) + 1))

    axes[-1].set_xlabel("Year")

    plt.suptitle("Yearly failure per fault type", y=1.02)
    plt.tight_layout()
    plt.show()

##################################################################
#             Wind mil data
##################################################################

kNOK_per_MW = 42678 * 1000
FX_rate = 0.093

EUR_per_MW = kNOK_per_MW * FX_rate

r= 0.07
horizon_years = 25
num_windmills = 1
turbine_capacity_mw = 2.5
utility_factor = 0.24
power_price_per_mwh = 125.11
capex = EUR_per_MW * turbine_capacity_mw

print(f"CAPEX: {capex}")

# fault_data = [
#     {
#         "name": "Gearbox",
#         "shape": 1.38,
#         "scale": 15.02,          # years
#         "DC": 647101,
#         "DT": 261, # 10 days
#     },
#     {
#         "name": "Generator",
#         "shape": 1.52,
#         "scale": 18.72,
#         "DC": 232634,
#         "DT": 126, # 7 days
#     },
#     {
#         "name": "Blade",
#         "shape": 0.75,
#         "scale": 86.80,
#         "DC": 374689,
#         "DT": 147, # 15 days
#     },
# ]


cost_data = {
    "Generator": {"DC": 232634, "DT": 126},
    "Gearbox": {"DC": 647101, "DT": 261},
    "Blade": {"DC": 374689, "DT": 147},
}

fault_data = []

for name, R_obs in observed_data.items():

    (k, lam), ess = fit_nhpp_per_year_prob(R_obs)

    fault_data.append({
        "name": name,
        "shape": k,
        "scale": lam,
        "DC": cost_data[name]["DC"],
        "DT": cost_data[name]["DT"]
    })

    print(f"{name} shape = {k} and scale = {lam}")


# Run Monte Carlo
mc = monte_carlo_lifecycle_cost(
    n_sims=10000,
    n_turbines = num_windmills,
    fault_data=fault_data,
    horizon_years=horizon_years,
    installed_capacity_mw_per_turbine=turbine_capacity_mw,
    utility_factor=utility_factor,
    power_price_per_mwh=power_price_per_mwh,
    discount_rate=r,
    capex_per_turbine=capex,
    seed=32,
)

# summary = yearly_failure_percentage(
#     records_values=mc["records_values"],
#     n_sims=10000,
#     horizon_years=20,
# )

# #print(summary)
# plot_yearly_failure_percentage(summary, observed_data)

all_events = pd.concat(mc["records_values"], ignore_index=True)
# Total cost and failures per sim_id and component
per_sim_component = (
    all_events
    .groupby(["sim_id", "fault_type"])
    .agg(
        n_failures=("discounted_cost", "size"),
        total_cost=("discounted_cost", "sum"),
    )
    .reset_index()
)

# Fill missing sim/component combos with zeros (important!)
fault_types = sorted(all_events["fault_type"].unique())
full_index = pd.MultiIndex.from_product(
    [range(len(mc["records_values"])), fault_types],
    names=["sim_id", "fault_type"]
)

per_sim_component = (
    per_sim_component
    .set_index(["sim_id", "fault_type"])
    .reindex(full_index, fill_value=0)
    .reset_index()
)

# Expected (mean) across simulations
component_summary = (
    per_sim_component
    .groupby("fault_type")
    .agg(
        expected_failures=("n_failures", "mean"),
        expected_cost=("total_cost", "mean"),
    )
    .reset_index()
)

print(component_summary)


# Results
losses = mc["lc_values"]
losses_pct = mc["lc_pct_capex_values"]
failures = mc["n_failures_values"]

mean_loss = losses.mean()
mean_fail = failures.mean()
median_loss = np.median(losses)
var95, cvar95 = var_cvar(losses, alpha=0.95)
var99, cvar99 = var_cvar(losses, alpha=0.99)

mean_loss_pct = losses_pct.mean()
var95_pct, cvar95_pct = var_cvar(losses_pct, alpha=0.95)

print("Lifecycle cost results")
print("----------------------")
print(f"Mean fail:   {mean_fail}")
print(f"Mean LC:     {mean_loss:,.0f}")
print(f"Median LC:   {median_loss:,.0f}")
print(f"VaR 95%:     {var95:,.0f}")
print(f"CVaR 95%:    {cvar95:,.0f}")
print(f"VaR 99%:     {var99:,.0f}")
print(f"CVaR 99%:    {cvar99:,.0f}")
print()
print("As % of CAPEX")
print("-------------")
print(f"Mean LC %:   {mean_loss_pct:.2f}%")
print(f"VaR 95% %:   {var95_pct:.2f}%")
print(f"CVaR 95% %:  {cvar95_pct:.2f}%")