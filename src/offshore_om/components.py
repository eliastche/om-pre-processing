"""
Wind-farm component and failure modelling.

This module defines the core physical entities used in the offshore wind
operation and maintenance model, including component types, vessels,
wind regions, turbines, and wind farms.

It also provides utilities for calibrating Weibull failure distributions
and constructing component definitions for Monte Carlo simulations.
"""

from dataclasses import dataclass
import numpy as np
from scipy.special import gamma
from copy import deepcopy
from .constants import kNOKEUR, kNOKGBP, HOURS_PER_YEAR
from .campaign import Failure

SEVERITIES = ["minor", "major", "replace"]

component_data = {
    "Generator": {
        "shape": 1.52,
        "scale": 18.72,
        "severity_probs": [0.5383, 0.3563, 0.1054],
        "global_failure_rate": 0.8778,
        "failure_rate_carrol": {"minor": 0.5378, "major": 0.3559, "replace": 0.008},
        "failure_rate_hendriks": {"minor": 0.049, "major": 0.018, "replace": 0.008},
        "failure_rate_jenkins_floating": {"minor": 0.5378, "major": 0.3559, "replace": 0.008},
        "failure_rate_jenkins_fixed": {"minor": 0.049, "major": 0.018, "replace": 0.008},
        "repair_time": {"minor": 7, "major": 24, "replace": 81},
        "material_cost": {"minor": 2667 * kNOKGBP, "major": 58333 * kNOKGBP, "replace": 1000000 * kNOKGBP},
        "technicians": {"minor": 2.2, "major": 2.7, "replace": 8},
    },
    "Gearbox": {
        "shape": 1.38,
        "scale": 15.02,
        "severity_probs": [0.6729, 0.0647, 0.2624],
        "global_failure_rate": 0.3350,
        "failure_rate_carrol": {"minor": 0.426, "major": 0.041, "replace": 0.042},
        "failure_rate_hendriks": {"minor": 0.644, "major": 0.157, "replace": 0.028},
        "failure_rate_jenkins_floating": {"minor": 0.5378, "major": 0.3559, "replace": 0.008},
        "failure_rate_jenkins_fixed": {"minor": 0.049, "major": 0.018, "replace": 0.008},
        "repair_time": {"minor": 8, "major": 22, "replace": 231},
        "material_cost": {"minor": 380 * kNOKGBP, "major": 7609 * kNOKGBP, "replace": 700000 * kNOKGBP},
        "technicians": {"minor": 2.2, "major": 3.2, "replace": 17},
    },
    "Blades": {
        "shape": 0.75,
        "scale": 86.8,
        "severity_probs": [0.9764, 0.0214, 0.0022],
        "global_failure_rate": 0.1732,
        "failure_rate_carrol": {"minor": 0.5078, "major": 0.0111, "replace": 0.0011},
        "failure_rate_hendriks": {"minor": 0.200, "major": 0.045, "replace": 0.040},
        "failure_rate_jenkins_floating": {"minor": 0.5378, "major": 0.3559, "replace": 0.008},
        "failure_rate_jenkins_fixed": {"minor": 0.049, "major": 0.018, "replace": 0.008},
        "repair_time": {"minor": 9, "major": 21, "replace": 288},
        "material_cost": {"minor": 819 * kNOKGBP, "major": 7222 * kNOKGBP, "replace": 433333 * kNOKGBP},
        "technicians": {"minor": 2.1, "major": 3.3, "replace": 21},
    },
}

vessel_data = {
    "minor": {
        "day_rate": 3000 * kNOKEUR,
        "mobilisation_cost": 0,
        "mobilisation_hours": 0,
        "speed_kmh": 37.04,
    },
    "major": {
        "day_rate": 12500 * kNOKEUR,
        "mobilisation_cost": 0,
        "mobilisation_hours": 504,
        "speed_kmh": 22.224,
    },
    "replace": {
        "day_rate": 360000 * kNOKGBP,
        "mobilisation_cost": 1800000 * kNOKGBP,
        "mobilisation_hours": 1440,
        "speed_kmh": 20.372,
    },
}

@dataclass
class ComponentType:
    """
    Definition of a wind-turbine component type.

    This object stores reliability, repair, workforce, and cost parameters
    used when simulating component failures and maintenance activities.

    Attributes
    ----------
    name : str
        Component name.
    shape : float
        Weibull shape parameter.
    scale : float
        Weibull scale parameter expressed in years.
    global_failure_rate : float
        Annual component failure rate.
    failure_rate : dict
        Failure rates grouped by severity category.
    severity_probs : list of float
        Relative probabilities of each severity category.
    material_cost : dict
        Material replacement costs by severity level.
    repair_time : dict
        Repair durations in hours by severity level.
    nr_workers : dict
        Number of technicians required by severity level.
    """
    name: str
    shape: float
    scale: float
    global_failure_rate: float
    failure_rate: dict
    severity_probs: list
    material_cost: dict
    repair_time: dict
    nr_workers: dict

@dataclass
class VesselType:
    """
    Definition of a vessel used for offshore maintenance campaigns.

    Attributes
    ----------
    name : str
        Vessel category.
    day_rate : float
        Daily charter rate.
    mobilisation_cost : float
        One-time mobilisation cost.
    mobilisation_hours : float
        Time required before the vessel becomes operational.
    speed_kmh : float
        Transit speed in kilometres per hour.
    """
    name: str
    day_rate: float
    mobilisation_cost: float
    mobilisation_hours: float
    speed_kmh: float

@dataclass
class WindRegion:
    """
    Offshore wind development region.

    Attributes
    ----------
    name : str
        Region name.
    capacity_factor : float
        Average wind-farm capacity factor.
    distance_to_shore_km : float
        Distance to the nearest service port in kilometres.
    floating : bool
        Whether the site uses floating foundations.
    """
    name: str
    capacity_factor: float
    distance_to_shore_km: float
    floating: bool


WindRegions = [
    WindRegion("Nordavind", 0.494, 200, floating=True),
    WindRegion("Nordvest", 0.483, 110, floating=True),
    WindRegion("Vestavind1", 0.489, 70, floating=True),
    WindRegion("Vestavind2", 0.512, 50, floating=True),
    WindRegion("SorvestA", 0.545, 122, floating=False),
    WindRegion("SorvestB", 0.543, 152, floating=False),
    WindRegion("SorvestC", 0.551, 153, floating=False),
    WindRegion("SorvestD", 0.545, 221, floating=False),
    WindRegion("SorvestE", 0.561, 112, floating=False),
    WindRegion("SorvestF", 0.559, 152, floating=False),
    WindRegion("Sonnavind", 0.565, 60, floating=True)
]

class Component:
    """
    Individual turbine component.

    A component tracks its remaining life and is responsible for
    sampling new lifetimes after replacement events.

    Parameters
    ----------
    comp_type : ComponentType
        Component definition.
    rng : numpy.random.Generator
        Random number generator.
    """
    def __init__(self, comp_type: ComponentType, rng: np.random.Generator):
        self.type = comp_type
        self.remaining_life_h = self.sample_life_h(rng)

    def sample_life_h(self, rng: np.random.Generator):
        """
        Draw a new component lifetime from the Weibull distribution.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        float
            Sampled component lifetime in hours.
        """
        return rng.weibull(self.type.shape) * self.type.scale * HOURS_PER_YEAR

    def age(self, dt_h: float):
        """
        Age the component.

        Parameters
        ----------
        dt_h : float
            Time increment in hours.
        """
        self.remaining_life_h -= dt_h

    def replace(self, rng: np.random.Generator):
        """
        Replace the component.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator.
        """
        self.remaining_life_h = self.sample_life_h(rng)

class Turbine:
    """
    Wind turbine with multiple components.

    Parameters
    ----------
    turbine_id : int
        Unique turbine identifier.
    component_types : list[ComponentType]
        List of component types for the turbine.
    rng : numpy.random.Generator
        Random number generator.
    """
    def __init__(self, turbine_id: int, component_types: list[ComponentType], rng: np.random.Generator):
        self.id = turbine_id
        self.components = [Component(comp_type, rng) for comp_type in component_types]
        self.down_since_h = None

    @property
    def operational(self):
        """
        Whether the turbine is currently operational.

        Returns
        -------
        bool
            True if the turbine is operational, otherwise False.
        """
        return self.down_since_h is None

    def age(self, dt_h: float):
        """
        Age all turbine components.

        Parameters
        ----------
        dt_h : float
            Time increment in hours.
        """
        if not self.operational:
            return
        for component in self.components:
            component.age(dt_h)

    def fail(self, time_h: float):
        """
        Mark the turbine as unavailable.

        Parameters
        ----------
        time_h : float
            Failure time in simulation hours.
        """
        if self.operational:
            self.down_since_h = time_h

    def repair(self):
        """
        Restore turbine operation after maintenance.
        """
        self.down_since_h = None

    def downtime_h(self, current_time_h: float):
        """
        Calculate accumulated downtime.

        Parameters
        ----------
        current_time_h : float
            Current simulation time in hours.

        Returns
        -------
        float
            Downtime duration in hours.
        """
        if self.down_since_h is None:
            return 0.0
        return current_time_h - self.down_since_h

class WindFarm:
    """
    Collection of wind turbines participating in the simulation.

    Parameters
    ----------
    n_turbines : int
        Number of turbines.
    component_types : list of ComponentType
        Component definitions installed in every turbine.
    rng : numpy.random.Generator
        Random number generator.
    """
    def __init__(self, n_turbines: int, component_types: list[ComponentType], rng: np.random.Generator):
        self.turbines = [Turbine(turbine_id, component_types, rng) for turbine_id in range(n_turbines)]

    def age(self, dt_h: float):
        """
        Age every turbine in the wind farm.

        Parameters
        ----------
        dt_h : float
            Time increment in hours.
        """
        for turbine in self.turbines:
            turbine.age(dt_h)

    def next_failure(self, current_time_h: float):
        """
        Determine the next component failure in the wind farm.

        Parameters
        ----------
        current_time_h : float
            Current simulation time in hours.

        Returns
        -------
        tuple
            Tuple containing:

            * failure_time_h : float
            * failure : Failure or None

        Notes
        -----
        The failure corresponds to the component with the shortest
        remaining lifetime among all operational turbines.
        """

        soonest_remaining_life_h = np.inf
        failed_turbine = None
        failed_component_id = None

        for turbine in self.turbines:
            if not turbine.operational:
                continue
            for component_id, component in enumerate(turbine.components):
                if component.remaining_life_h < soonest_remaining_life_h:
                    soonest_remaining_life_h = component.remaining_life_h
                    failed_turbine = turbine
                    failed_component_id = component_id

        if failed_turbine is None:
            return np.inf, None

        soonest_remaining_life_h = max(0.0, soonest_remaining_life_h)
        failure_time_h = current_time_h + soonest_remaining_life_h
        component = failed_turbine.components[failed_component_id]

        failure = Failure(
            turbine_id=failed_turbine.id,
            component_id=failed_component_id,
            component_name=component.type.name,
            severity="replace",
            failure_time_h=failure_time_h,
        )

        return failure_time_h, failure

    def register_failure(self, failure: Failure):
        """
        Register a failure and stop the affected turbine.

        Parameters
        ----------
        failure : Failure
            Failure event to register.
        """
        turbine = self.turbines[failure.turbine_id]
        turbine.fail(failure.failure_time_h)

    def repair_failure(self, failure: Failure, rng: np.random.Generator):
        """
        Repair a failed component and restart the turbine.

        Parameters
        ----------
        failure : Failure
            Failure being repaired.
        rng : numpy.random.Generator
            Random number generator used to sample the replacement life.
        """
        turbine = self.turbines[failure.turbine_id]
        component = turbine.components[failure.component_id]
        component.replace(rng)
        turbine.repair()

def calibrated_replacement_scale(component: ComponentType):
    """
    Calculate the Weibull scale parameter required to match a target
    replacement failure rate.

    Parameters
    ----------
    component : ComponentType
        Component definition.

    Returns
    -------
    float
        Calibrated Weibull scale parameter.

    Raises
    ------
    ValueError
        If the replacement failure rate is non-positive.

    Notes
    -----
    The calibration uses the analytical Weibull mean time between failures.
    """
    target_rate = component.failure_rate["replace"]
    if target_rate <= 0:
        raise ValueError(f"Replacement failure rate must be positive for {component.name}")
    target_mtbf = 1.0 / target_rate
    return target_mtbf / gamma(1.0 + 1.0 / component.shape)

def build_component_types(
    capacity_per_turbine: float,
    failure_rate_type: str,
    floating: bool,
):
    """
    Construct component definitions for a wind farm.

    Material costs are scaled according to turbine rating and Weibull
    scale parameters are recalibrated to match the selected replacement
    failure rates.

    Parameters
    ----------
    capacity_per_turbine : float
        Turbine rating in MW.
    failure_rate_type : str
        Failure-rate dataset to use. Supported values include
        ``"carrol"``, ``"hendriks"``, and ``"jenkins"``.
    floating : bool
        Whether floating or fixed-bottom assumptions should be used.

    Returns
    -------
    list of ComponentType
        Component definitions for blades, gearbox, and generator.

    Notes
    -----
    When ``failure_rate_type`` is ``"jenkins"``, separate floating and
    fixed-bottom calibrations are automatically selected.
    """
    material_cost_scale = capacity_per_turbine / 10.0
    comps = []
    if failure_rate_type == "jenkins":
        if floating:
            failure_rate_type = "jenkins_floating"
        else:
            failure_rate_type = "jenkins_fixed"

    for name in ["Blades", "Gearbox", "Generator"]:
        data = component_data[name]
        comp = ComponentType(
            name=name,
            shape=data["shape"],
            scale=data["scale"],
            global_failure_rate=data["global_failure_rate"],
            failure_rate=data[f"failure_rate_{failure_rate_type}"],
            severity_probs=data["severity_probs"],
            material_cost={sev: material_cost_scale * val for sev, val in data["material_cost"].items()},
            repair_time=deepcopy(data["repair_time"]),
            nr_workers=deepcopy(data["technicians"]),
        )
        comp.scale = calibrated_replacement_scale(comp)
        comps.append(comp)
    return comps
