"""

Simulation utilities for modeling offshore wind O&M processes.

This module provides classes and functions for simulating failure, maintenance, and repair activities in offshore wind farms.
"""

from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from offshore_om.components import WindRegion, WindFarm, vessel_data
from offshore_om.constants import HOURS_PER_YEAR
from offshore_om.downtime import CostTracker
from offshore_om.campaign import Campaign, Failure


@dataclass
class SimulationSettings:
    """
    Configuration parameters for an offshore wind O&M simulation.

    The settings define the simulation horizon, workforce schedule, turbine
    capacity, campaign parts capacity, and resupply duration.

    Attributes
    ----------
    horizon_years : float
        Duration of the simulation horizon in years.
    shift_hours : float
        Number of working hours in one crew shift.
    daily_rate : float
        Daily labour cost per worker.
    capacity_per_turbine : float
        Rated generation capacity of one turbine in MW.
    part_capacity : int
        Maximum number of replacement parts carried by a campaign.
    resupply_time_h : float
        Time required to reload replacement parts at port, in hours.
    """
    horizon_years: float
    shift_hours: float
    daily_rate: float
    capacity_per_turbine: float
    part_capacity: int
    resupply_time_h: float


class Simulation:
    """
    Simulate failure, maintenance, and repair campaigns for an offshore wind farm.

    The simulation advances through failure events, groups failures into
    maintenance campaigns, applies vessel and labour constraints, and records
    repair costs, downtime, and lost energy production.

    Parameters
    ----------
    farm : WindFarm
        Wind farm containing the turbines and components to simulate.
    region : WindRegion
        Region containing geographic and production characteristics.
    settings : SimulationSettings
        Operational and economic simulation settings.
    rng : numpy.random.Generator
        Random number generator used by stochastic simulation processes.

    Attributes
    ----------
    farm : WindFarm
        Wind farm being simulated.
    region : WindRegion
        Offshore wind region used by the simulation.
    settings : SimulationSettings
        Simulation configuration.
    rng : numpy.random.Generator
        Random number generator used by the simulation.
    t_h : float
        Current simulation time in hours.
    costs : CostTracker
        Accumulator for cost, downtime, and lost-production events.
    n_campaigns : int
        Number of maintenance campaigns initiated.
    n_failures : int
        Number of failures registered.
    n_repairs : int
        Number of repairs completed.
    repairs_by_component : collections.defaultdict
        Number of completed repairs grouped by component name.
    failures_by_component : collections.defaultdict
        Number of registered failures grouped by component name.
    pending_failures : list
        Failures deferred until the current frozen campaign is completed.

    Notes
    -----
    When ``DEBUG`` is enabled, simulation events are printed with their
    corresponding simulation time.
    """
    DEBUG = False

    def __init__(
        self,
        farm: WindFarm,
        region: WindRegion,
        settings: SimulationSettings,
        rng: np.random.Generator,
    ):
        """
        Initialize an offshore wind O&M simulation.

        Parameters
        ----------
        farm : WindFarm
            Wind farm containing the turbines and components to simulate.
        region : WindRegion
            Region containing distance and capacity-factor information.
        settings : SimulationSettings
            Operational and economic simulation settings.
        rng : numpy.random.Generator
            Random number generator used by stochastic model components.
        """
                
        self.farm = farm
        self.region = region
        self.settings = settings
        self.rng = rng

        self.t_h = 0.0
        self.costs = CostTracker()

        self.n_campaigns = 0
        self.n_failures = 0
        self.n_repairs = 0
        self.repairs_by_component = defaultdict(int)
        self.failures_by_component = defaultdict(int)

        self.pending_failures = []

    def log(self, *args):
        """
        Print a timestamped simulation message when debugging is enabled.

        Parameters
        ----------
        *args
            Values passed to :func:`print`.

        Notes
        -----
        The timestamp is expressed in simulation years.
        """

        if self.DEBUG:
            print(f"[{self.t_h / HOURS_PER_YEAR:7.3f} yr]", *args)

    def advance_frozen_campaign_activity(
        self,
        duration_h: float,
        horizon_h: float,
    ):
        """
        Advance time during an activity whose campaign scope is frozen.

        Failures occurring during the activity are registered and added to the
        pending-failure queue rather than to the active campaign. If the
        activity extends beyond the simulation horizon, the remaining time is
        advanced without aging the wind farm.

        Parameters
        ----------
        duration_h : float
            Duration of the campaign activity in hours.
        horizon_h : float
            End of the simulation horizon in hours.
        """

        activity_end_h = self.t_h + duration_h
        search_until_h = min(activity_end_h, horizon_h)

        while self.t_h < search_until_h:
            next_failure_h, new_failure = self.farm.next_failure(self.t_h)

            if new_failure is None or next_failure_h > search_until_h:
                self.advance_to(search_until_h)
                break

            self.advance_to(next_failure_h)
            self.farm.register_failure(new_failure)

            self.n_failures += 1
            self.failures_by_component[new_failure.component_name] += 1
            self.pending_failures.append(new_failure)

            self.log(
                "FAILURE DURING FROZEN CAMPAIGN - DEFERRED",
                f"WT={new_failure.turbine_id}",
                f"Component={new_failure.component_name}",
                f"FailureTime={new_failure.failure_time_h / HOURS_PER_YEAR:.3f} yr",
                f"Pending={len(self.pending_failures)}",
            )

        if self.t_h < activity_end_h:
            self.jump_to_without_aging(activity_end_h)

    def advance_to(self, new_time_h: float):
        """
        Advance the simulation clock and age the wind farm.

        Parameters
        ----------
        new_time_h : float
            Target simulation time in hours.

        Raises
        ------
        ValueError
            If the target time is earlier than the current simulation time,
            except for differences smaller than the numerical tolerance.
        """

        if new_time_h < self.t_h:
            if self.t_h - new_time_h < 1e-7:
                return
            raise ValueError(
                f"Simulation time cannot move backwards: "
                f"current={self.t_h}, new={new_time_h}"
            )

        dt_h = new_time_h - self.t_h
        if dt_h > 0:
            self.farm.age(dt_h)
            self.t_h = new_time_h

    def one_way_transit_h(self, vessel: dict):
        """
        Calculate the vessel's one-way transit time to or from the wind farm.

        Parameters
        ----------
        vessel : dict
            Vessel data containing the ``speed_kmh`` entry.

        Returns
        -------
        float
            One-way transit time in hours.
        """

        return self.region.distance_to_shore_km / vessel["speed_kmh"]

    def jump_to_without_aging(self, new_time_h: float):
        """
        Move the simulation clock without aging wind-farm components.

        Parameters
        ----------
        new_time_h : float
            Target simulation time in hours.

        Raises
        ------
        ValueError
            If the target time is earlier than the current simulation time,
            except for differences smaller than the numerical tolerance.

        Notes
        -----
        This method changes the simulation clock without updating component
        remaining lifetimes.
        """

        if new_time_h < self.t_h:
            if self.t_h - new_time_h < 1e-7:
                return
            raise ValueError(
                f"Simulation time cannot move backwards: "
                f"current={self.t_h}, new={new_time_h}"
            )

        self.t_h = new_time_h

    def run(self, horizon_years: float = None):
        """
        Run the failure and maintenance simulation.

        The simulation processes pending failures first. Otherwise, it advances
        to the next wind-farm failure. Each initiating failure creates a new
        maintenance campaign, after which campaign execution is delegated to
        :meth:`run_campaign`.

        Parameters
        ----------
        horizon_years : float, optional
            Simulation horizon in years. If omitted, the value from
            :attr:`SimulationSettings.horizon_years` is used.
        """
                
        if horizon_years is None:
            horizon_years = self.settings.horizon_years

        horizon_h = horizon_years * HOURS_PER_YEAR

        while self.t_h < horizon_h or self.pending_failures:
            if self.pending_failures:
                failure = self.pending_failures.pop(0)
                self.log(
                    "PENDING FAILURE STARTS NEW CAMPAIGN",
                    f"WT={failure.turbine_id}",
                    f"Component={failure.component_name}",
                    f"FailureTime={failure.failure_time_h / HOURS_PER_YEAR:.3f} yr",
                    f"CampaignStart={self.t_h / HOURS_PER_YEAR:.3f} yr",
                )
            else:
                failure_time_h, failure = self.farm.next_failure(self.t_h)
                if failure is None:
                    break

                if failure_time_h > horizon_h:
                    break

                self.advance_to(failure_time_h)
                self.farm.register_failure(failure)

                self.n_failures += 1
                self.failures_by_component[failure.component_name] += 1

                self.log(
                    "FAILURE STARTS NEW CAMPAIGN",
                    f"WT={failure.turbine_id}",
                    f"Component={failure.component_name}",
                    f"Severity={failure.severity}",
                    f"FailureTime={failure.failure_time_h / HOURS_PER_YEAR:.3f} yr",
                )

            campaign = Campaign(
                vessel=vessel_data["replace"],
                start_time_h=self.t_h,
                parts_capacity=self.settings.part_capacity,
            )
            campaign.add_failure(failure)
            self.n_campaigns += 1

            self.log(
                "CAMPAIGN REQUESTED",
                f"MobilisationHours={campaign.vessel['mobilisation_hours']:.1f}",
                f"Arrival={campaign.arrival_time_h / HOURS_PER_YEAR:.3f} yr",
            )

            self.run_campaign(campaign, horizon_h)

    def run_campaign(self, campaign: Campaign, horizon_h: float):
        """
        Execute a complete maintenance campaign.

        The campaign collects failures during vessel mobilisation, freezes its
        repair scope upon vessel arrival, applies weather and transit delays,
        repairs the frozen backlog, performs resupply when required, and
        returns the vessel to port. Campaign labour and vessel costs are added
        after the vessel returns.

        Parameters
        ----------
        campaign : Campaign
            Maintenance campaign to execute.
        horizon_h : float
            End of the simulation horizon in hours.
        """

        self.collect_failures_until(
            end_time_h=min(campaign.arrival_time_h, horizon_h),
            campaign=campaign,
            horizon_h=horizon_h,
        )

        if self.t_h < campaign.arrival_time_h:
            if campaign.arrival_time_h <= horizon_h:
                self.advance_to(campaign.arrival_time_h)
            else:
                self.jump_to_without_aging(campaign.arrival_time_h)

        campaign.charter_start_h = self.t_h
        frozen_backlog_size = len(campaign.failures)
        campaign.crew_workers = self.campaign_crew_workers(campaign.failures)

        weather_delay_h = campaign.weather_delay_hours(arrival_time_h=self.t_h, rng=self.rng)

        self.log(
            f"VESSEL ARRIVED - CAMPAIGN SCOPE FROZEN - WEATHER DELAY {weather_delay_h:.1f} hrs",
            f"Backlog={frozen_backlog_size}",
            f"CrewWorkers={campaign.crew_workers}",
            f"CharterStart={campaign.charter_start_h / HOURS_PER_YEAR:.3f} yr",
        )

        self.advance_frozen_campaign_activity(duration_h=weather_delay_h, horizon_h=horizon_h)

        transit_out_h = self.one_way_transit_h(campaign.vessel)
        self.log("SAIL TO WIND FARM", f"TransitHours={transit_out_h:.1f}")

        self.advance_frozen_campaign_activity(duration_h=transit_out_h, horizon_h=horizon_h)
        campaign.vessel_at_farm = True

        while campaign.has_backlog():
            failure = campaign.pop_next_failure()
            if campaign.needs_resupply():
                self.perform_resupply(campaign, horizon_h)

            if not campaign.vessel_at_farm:
                transit_h = self.one_way_transit_h(campaign.vessel)
                self.log(
                    "SAIL TO WIND FARM AFTER RESUPPLY",
                    f"TransitHours={transit_h:.1f}",
                )
                self.advance_frozen_campaign_activity(duration_h=transit_h, horizon_h=horizon_h)
                campaign.vessel_at_farm = True

            self.repair_one_failure_with_campaign_logic(
                failure=failure,
                campaign=campaign,
                horizon_h=horizon_h,
            )

        if campaign.vessel_at_farm:
            transit_home_h = self.one_way_transit_h(campaign.vessel)
            self.log("SAIL BACK TO PORT", f"TransitHours={transit_home_h:.1f}")
            self.advance_frozen_campaign_activity(duration_h=transit_home_h, horizon_h=horizon_h)

        campaign.vessel_at_farm = False
        campaign.vessel_returned_to_port = True
        campaign.charter_end_h = self.t_h
        self.add_campaign_costs(campaign)

        self.log(
            "CAMPAIGN END",
            f"Campaigns={self.n_campaigns}",
            f"Failures={self.n_failures}",
            f"Repairs={self.n_repairs}",
            f"Pending={len(self.pending_failures)}",
            f"CharterEnd={campaign.charter_end_h / HOURS_PER_YEAR:.3f} yr",
            f"DirectCost={self.costs.total_direct_cost:.2f}",
        )

    def perform_resupply(self, campaign: Campaign, horizon_h: float):
        """
        Return a campaign to port and replenish its replacement parts.

        The activity consists of transit to port followed by the configured
        loading period. Failures occurring during these activities are placed
        in the pending-failure queue.

        Parameters
        ----------
        campaign : Campaign
            Active maintenance campaign requiring replacement parts.
        horizon_h : float
            End of the simulation horizon in hours.
        """

        vessel = campaign.vessel
        sail_to_port_h = self.one_way_transit_h(vessel)
        loading_h = self.settings.resupply_time_h

        self.log(
            "RESUPPLY START",
            f"SailToPortHours={sail_to_port_h:.1f}",
            f"LoadingHours={loading_h:.1f}",
        )

        self.advance_frozen_campaign_activity(duration_h=sail_to_port_h, horizon_h=horizon_h)
        self.advance_frozen_campaign_activity(duration_h=loading_h, horizon_h=horizon_h)
        campaign.resupply()

        self.log(
            "RESUPPLY COMPLETE",
            f"PartsRemaining={campaign.parts_remaining}",
        )

    def collect_failures_until(
        self,
        end_time_h: float,
        campaign: Campaign,
        horizon_h: float,
    ):
        """
        Collect failures occurring while a campaign is mobilising.

        The simulation advances through failure events until the requested end
        time or the simulation horizon is reached. Each detected failure is
        registered with the wind farm and added to the campaign backlog.

        Parameters
        ----------
        end_time_h : float
            Time until which failures should be collected, in hours.
        campaign : Campaign
            Mobilising campaign receiving the failures.
        horizon_h : float
            End of the simulation horizon in hours.
        """

        end_time_h = min(end_time_h, horizon_h)

        while self.t_h < end_time_h:
            next_failure_h, failure = self.farm.next_failure(self.t_h)
            if failure is None or next_failure_h > end_time_h:
                self.advance_to(end_time_h)
                return

            if next_failure_h < self.t_h:
                next_failure_h = self.t_h

            self.advance_to(next_failure_h)
            self.farm.register_failure(failure)

            self.n_failures += 1
            self.failures_by_component[failure.component_name] += 1
            campaign.add_failure(failure)

            self.log(
                "ADDED TO MOBILISING CAMPAIGN",
                f"WT={failure.turbine_id}",
                f"Component={failure.component_name}",
                f"Backlog={len(campaign.failures)}",
            )

    def repair_one_failure_with_campaign_logic(
        self,
        failure: Failure,
        campaign: Campaign,
        horizon_h: float,
    ):
        """
        Repair one failure using the active campaign's resources.

        The method determines the component-specific repair duration, advances
        frozen campaign time, consumes one replacement part, completes the
        repair, and updates repair counters.

        Parameters
        ----------
        failure : Failure
            Failure to repair.
        campaign : Campaign
            Campaign performing the repair.
        horizon_h : float
            End of the simulation horizon in hours.
        """

        turbine = self.farm.turbines[failure.turbine_id]
        component = turbine.components[failure.component_id]
        comp_type = component.type

        repair_time_h = comp_type.repair_time[failure.severity]

        self.log(
            "REPAIR START",
            f"WT={failure.turbine_id}",
            f"Component={failure.component_name}",
            f"RepairHours={repair_time_h:.1f}",
            f"PartsBefore={campaign.parts_remaining}",
        )

        self.advance_frozen_campaign_activity(duration_h=repair_time_h, horizon_h=horizon_h)
        campaign.use_part()

        self.complete_repair(failure)
        self.n_repairs += 1
        self.repairs_by_component[failure.component_name] += 1

        self.log(
            "REPAIR COMPLETE",
            f"WT={failure.turbine_id}",
            f"Component={failure.component_name}",
            f"PartsAfter={campaign.parts_remaining}",
        )

    def complete_repair(self, failure: Failure):
        """
        Complete a repair and record its cost and production-loss consequences.

        The method calculates turbine downtime, material cost, and lost energy
        production. It records the resulting event in the cost tracker and
        restores the failed component through the wind-farm model.

        Parameters
        ----------
        failure : Failure
            Failure whose repair has been completed.
        """

        turbine = self.farm.turbines[failure.turbine_id]
        component = turbine.components[failure.component_id]
        comp_type = component.type

        downtime_h = turbine.downtime_h(self.t_h)
        material_cost = comp_type.material_cost[failure.severity]

        t_years = self.t_h / HOURS_PER_YEAR

        downtime_start_h = self.t_h - downtime_h
        downtime_end_h = self.t_h

        downtime_start_years = downtime_start_h / HOURS_PER_YEAR
        downtime_end_years = downtime_end_h / HOURS_PER_YEAR

        lost_mwh = (
            downtime_h
            * self.settings.capacity_per_turbine
            * self.region.capacity_factor
        )

        self.costs.add_event(
            time_years=t_years,
            component_name=failure.component_name,
            downtime_hours=downtime_h,
            lost_mwh=lost_mwh,
            material_cost=material_cost,
            labour_cost=0.0,
            transport_cost=0.0,
            downtime_start_years=downtime_start_years,
            downtime_end_years=downtime_end_years,
        )

        self.log(
            "REPAIR COST",
            f"Component={failure.component_name}",
            f"MaterialCost={material_cost:.2f}",
            f"DowntimeHours={downtime_h:.1f}",
            f"LostMWh={lost_mwh:.2f}",
        )

        self.farm.repair_failure(failure, self.rng)

    def add_campaign_costs(self, campaign: Campaign):
        """
        Calculate and record labour and vessel costs for a campaign.

        Campaign duration is converted to billable days by rounding upward.
        Vessel costs include mobilisation and daily charter costs. Labour costs
        depend on crew size, billable days, and the configured daily rate.

        Parameters
        ----------
        campaign : Campaign
            Completed campaign for which costs are calculated.

        Raises
        ------
        ValueError
            If the campaign charter start or end time has not been set.
        """

        vessel = campaign.vessel

        if campaign.charter_start_h is None or campaign.charter_end_h is None:
            raise ValueError("Campaign charter start/end has not been set.")

        campaign_duration_h = max(0.0, campaign.charter_end_h - campaign.charter_start_h)
        billable_campaign_days = np.ceil(campaign_duration_h / 24.0)

        vessel_cost = vessel["mobilisation_cost"] + vessel["day_rate"] * billable_campaign_days
        labour_cost = campaign.crew_workers * billable_campaign_days * self.settings.daily_rate

        t_years = campaign.charter_start_h / HOURS_PER_YEAR

        self.costs.add_event(
            time_years=t_years,
            component_name="Campaign labour and vessel",
            downtime_hours=0.0,
            lost_mwh=0.0,
            material_cost=0.0,
            labour_cost=labour_cost,
            transport_cost=vessel_cost,
        )

        self.log(
            "CAMPAIGN COST",
            f"CampaignDurationHours={campaign_duration_h:.1f}",
            f"BillableCampaignDays={billable_campaign_days:.0f}",
            f"CrewWorkers={campaign.crew_workers}",
            f"LabourCost={labour_cost:.2f}",
            f"VesselCost={vessel_cost:.2f}",
        )

    def component_type_from_failure(self, failure: Failure):
        """
        Retrieve the component type associated with a failure.

        Parameters
        ----------
        failure : Failure
            Failure identifying a turbine and component.

        Returns
        -------
        object
            Component type assigned to the failed component.
        """

        turbine = self.farm.turbines[failure.turbine_id]
        component = turbine.components[failure.component_id]
        return component.type

    def campaign_crew_workers(self, failures: list[Failure]):
        """
        Determine the number of workers required for a campaign.

        The crew requirement is based on the largest repair workforce among
        the campaign failures. It is multiplied by the number of shifts needed
        to cover a 24-hour operating day and rounded upward. We use 24 hours 
        shift length as baseline, and the actual shift length is configurable 
        in the simulation settings.

        Parameters
        ----------
        failures : list of Failure
            Failures included in the campaign's frozen repair scope.

        Returns
        -------
        int
            Required number of campaign workers. Returns zero when the failure
            list is empty.
        """

        if not failures:
            return 0

        max_workers = 0.0
        for failure in failures:
            comp_type = self.component_type_from_failure(failure)
            workers = comp_type.nr_workers[failure.severity]
            max_workers = max(max_workers, workers)

        shift_multiplier = 24.0 / self.settings.shift_hours
        return int(np.ceil(max_workers * shift_multiplier))

    def print_status(self):
        """
        Print the current wind-farm and simulation status.

        The output includes turbine operating states, component remaining
        lifetimes, direct cost per repair, lost-production cost per repair,
        and transport cost per campaign.

        Notes
        -----
        This method writes directly to standard output.
        """

        print("\nFarm status")
        print("-" * 60)

        for turbine in self.farm.turbines:
            state = "UP" if turbine.operational else "DOWN"
            print(f"WT {turbine.id:02d}", state)
            for component in turbine.components:
                print(
                    f"  {component.type.name:<12}",
                    f"{component.remaining_life_h / HOURS_PER_YEAR:8.2f} years",
                )

        print("-" * 60)
        print("Direct cost per repair:", self.costs.total_direct_cost / self.n_repairs)
        print("Lost MWh per repair by valuation:")
        for valuation_name, value in self.costs.total_lost_prod_costs_by_valuation.items():
            print(f"  {valuation_name}: {value / self.n_repairs}")
        print("Transport cost per campaign:", self.costs.total_transport_cost / self.n_campaigns)

    @property
    def summary(self):
        """
        Return aggregate results for the current simulation.

        Returns
        -------
        dict
            Dictionary containing campaign, failure, and repair counts together
            with downtime, lost energy, and direct-cost totals. The cost totals
            are separated into material, labour, and transport costs.
        """

        return {
            "campaigns": self.n_campaigns,
            "failures": self.n_failures,
            "repairs": self.n_repairs,
            "downtime": self.costs.total_downtime_hours,
            "lost_mwh": self.costs.total_lost_mwh,
            "direct": self.costs.total_direct_cost,
            "material": self.costs.total_material_cost,
            "labour": self.costs.total_labour_cost,
            "transport": self.costs.total_transport_cost,
        }

def print_array_summary(name, values, unit=""):
    """
    Print descriptive statistics for an array of simulation results.

    Parameters
    ----------
    name : str
        Name displayed before the statistics.
    values : array-like
        Numerical values from which the statistics are calculated.
    unit : str, optional
        Unit displayed after the statistics.

    Notes
    -----
    The printed statistics are the mean and the 50th, 75th, and 95th
    percentiles.
    """

    values = np.asarray(values)
    print(
        f"{name:<35}"
        f"mean={np.mean(values):>12,.2f} "
        f"p50={np.quantile(values, 0.50):>12,.2f} "
        f"p75={np.quantile(values, 0.75):>12,.2f} "
        f"p95={np.quantile(values, 0.95):>12,.2f} "
        f"{unit}"
    )


def simulate_wind_farm_OandM(
    n_turbines: int,
    component_types: list = None,
    region: WindRegion = None,
    horizon_years: float = 25.0,
    capacity_per_turbine: float = 15.0,
    daily_rate: float = 10000.0,
    n_simulations: int = 10000,
    random_seed: int = None,
    shift_hours: float = 12,
    part_capacity: int = 3,
    resupply_time_h: float = 24,
    debug: bool = False,
    debug_sim: int = 0,
    progress_every: int = 1000,
):
    """
    Run repeated offshore wind O&M simulations.

    For each realization, the function creates a wind farm, initializes the
    simulation settings, executes the specified simulation horizon, and stores
    campaign, repair, downtime, cost, and lost-production results.

    Parameters
    ----------
    n_turbines : int
        Number of turbines in the simulated wind farm.
    component_types : list, optional
        Component-type definitions assigned to each turbine.
    region : WindRegion, optional
        Offshore wind region containing distance and production information.
    horizon_years : float, optional
        Simulation horizon in years. The default is 25 years.
    capacity_per_turbine : float, optional
        Rated capacity of one turbine in MW. The default is 15 MW.
    daily_rate : float, optional
        Daily labour cost per worker. The default is 10000.
    n_simulations : int, optional
        Number of Monte Carlo realizations. The default is 10000.
    random_seed : int, optional
        Seed used to initialize the random number generator. If omitted, NumPy
        initializes the generator without a fixed seed.
    shift_hours : float, optional
        Duration of one crew shift in hours. The default is 12 hours.
    part_capacity : int, optional
        Maximum number of replacement parts carried by a campaign. The default
        is 3.
    resupply_time_h : float, optional
        Time required to reload parts at port, in hours. The default is 24
        hours.
    debug : bool, optional
        Whether to enable detailed output for one simulation realization.
    debug_sim : int, optional
        Zero-based index of the realization for which debugging is enabled.
    progress_every : int, optional
        Number of simulations between progress messages. Set to ``None`` to
        disable periodic progress output.

    Returns
    -------
    dict
        Simulation inputs, event records, and aggregated outputs. Returned
        results include lifetime direct costs, lost energy, downtime,
        campaigns per realization, total repairs by component, installed
        capacity, and summary statistics.

    Raises
    ------
    ValueError
        If ``component_types`` is not provided.
    ValueError
        If ``region`` is not provided.

    Notes
    -----
    Summary statistics include the mean, 75th percentile,
    and upper-tail VaR and CVaR at the 95th-percentile threshold.
    """

    rng = np.random.default_rng(seed=random_seed)

    if component_types is None:
        raise ValueError("component_types must be provided.")

    if region is None:
        raise ValueError("region must be provided as a WindRegion instance.")

    total_direct_costs = np.zeros(n_simulations)
    total_downtime = np.zeros(n_simulations)
    total_lost_mwh = np.zeros(n_simulations)
    campaigns_per_simulation = np.zeros(n_simulations, dtype=int)

    events_by_simulation = []
    total_repairs_by_component = defaultdict(int)

    for sim in range(n_simulations):
        farm = WindFarm(
            n_turbines=n_turbines,
            component_types=component_types,
            rng=rng,
        )

        settings = SimulationSettings(
            horizon_years=horizon_years,
            shift_hours=shift_hours,
            daily_rate=daily_rate,
            capacity_per_turbine=capacity_per_turbine,
            part_capacity=part_capacity,
            resupply_time_h=resupply_time_h,
        )

        simulation = Simulation(
            farm=farm,
            region=region,
            settings=settings,
            rng=rng,
        )

        simulation.DEBUG = debug and sim == debug_sim
        simulation.sim_id = sim

        if simulation.DEBUG:
            print("\n" + "=" * 80)
            print(f"DEBUGGING SIMULATION {sim}")
            print("=" * 80)

        simulation.run(horizon_years)

        if simulation.DEBUG:
            simulation.print_status()

        if not debug and progress_every is not None:
            if (sim + 1) % progress_every == 0 or sim == n_simulations - 1 or sim == 0:
                print("=" * 80)
                print(f"\nCompleted simulation {sim + 1}/{n_simulations}")
                print("Campaigns:", simulation.n_campaigns)
                print("Repairs:", simulation.n_repairs)

        total_direct_costs[sim] = simulation.costs.total_direct_cost
        total_downtime[sim] = simulation.costs.total_downtime_hours
        total_lost_mwh[sim] = simulation.costs.total_lost_mwh
        campaigns_per_simulation[sim] = simulation.n_campaigns

        events_by_simulation.append(list(simulation.costs.events))

        for comp, count in simulation.repairs_by_component.items():
            total_repairs_by_component[comp] += count

    print("\nSIMULATION SUMMARY")
    print("Mean Campaigns:", np.mean(campaigns_per_simulation))
    print("Mean direct cost:", np.mean(total_direct_costs))
    print("Mean lost MWh:", np.mean(total_lost_mwh))
    print("Installed capacity:", n_turbines * capacity_per_turbine)
    print("Mean downtime per turbine:", np.mean(total_downtime) / n_turbines)

    print("\nAVERAGE REPAIRS PER TURBINE OVER HORIZON")
    for comp, total_repairs in total_repairs_by_component.items():
        mean_repairs = total_repairs / (n_simulations * n_turbines)
        print(f"{comp:10s}{mean_repairs:.3f} repairs/lifetime")

    def summary_stats(values):
        """
        Calculate summary and upper-tail statistics.

        Parameters
        ----------
        values : array-like
            Numerical simulation results.

        Returns
        -------
        dict
            Mean, 75th percentile, 95th percentile, and CVaR of observations
            greater than or equal to the 95th-percentile threshold.
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

    return {
        "n_turbines": n_turbines,
        "capacity_per_turbine": capacity_per_turbine,
        "installed_capacity": n_turbines * capacity_per_turbine,
        "horizon_years": horizon_years,

        "events_by_simulation": events_by_simulation,

        "lifetime_direct_costs": total_direct_costs,
        "lifetime_direct_stats": summary_stats(total_direct_costs),

        "lifetime_lost_mwh": total_lost_mwh,
        "lifetime_lost_mwh_stats": summary_stats(total_lost_mwh),

        "total_downtime_hours": total_downtime,
        "downtime": summary_stats(total_downtime),

        "campaigns_per_simulation": campaigns_per_simulation,
        "repairs_by_component_total": dict(total_repairs_by_component),
    }
