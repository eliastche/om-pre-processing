from dataclasses import dataclass
import numpy as np
from scipy.special import gamma
from copy import deepcopy
from .constants import kNOKEUR, kNOKGBP, HOURS_PER_YEAR
from .failures import Failure

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
    name: str
    day_rate: float
    mobilisation_cost: float
    mobilisation_hours: float
    speed_kmh: float

@dataclass
class WindRegion:
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
    def __init__(self, comp_type: ComponentType, rng: np.random.Generator):
        self.type = comp_type
        self.remaining_life_h = self.sample_life_h(rng)

    def sample_life_h(self, rng: np.random.Generator):
        return rng.weibull(self.type.shape) * self.type.scale * HOURS_PER_YEAR

    def age(self, dt_h: float):
        self.remaining_life_h -= dt_h

    def replace(self, rng: np.random.Generator):
        self.remaining_life_h = self.sample_life_h(rng)

class Turbine:
    def __init__(self, turbine_id: int, component_types: list[ComponentType], rng: np.random.Generator):
        self.id = turbine_id
        self.components = [Component(comp_type, rng) for comp_type in component_types]
        self.down_since_h = None

    @property
    def operational(self):
        return self.down_since_h is None

    def age(self, dt_h: float):
        if not self.operational:
            return
        for component in self.components:
            component.age(dt_h)

    def fail(self, time_h: float):
        if self.operational:
            self.down_since_h = time_h

    def repair(self):
        self.down_since_h = None

    def downtime_h(self, current_time_h: float):
        if self.down_since_h is None:
            return 0.0
        return current_time_h - self.down_since_h

class WindFarm:
    def __init__(self, n_turbines: int, component_types: list[ComponentType], rng: np.random.Generator):
        self.turbines = [Turbine(turbine_id, component_types, rng) for turbine_id in range(n_turbines)]

    def age(self, dt_h: float):
        for turbine in self.turbines:
            turbine.age(dt_h)

    def next_failure(self, current_time_h: float):
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

    def register_failure(self, failure: "Failure"):
        turbine = self.turbines[failure.turbine_id]
        turbine.fail(failure.failure_time_h)

    def repair_failure(self, failure: "Failure", rng: np.random.Generator):
        turbine = self.turbines[failure.turbine_id]
        component = turbine.components[failure.component_id]
        component.replace(rng)
        turbine.repair()

def calibrated_replacement_scale(component: ComponentType):
    target_rate = component.failure_rate["replace"]
    if target_rate <= 0:
        raise ValueError(f"Replacement failure rate must be positive for {component.name}")
    target_mtbf = 1.0 / target_rate
    return target_mtbf / gamma(1.0 + 1.0 / component.shape)

def build_component_types(capacity_per_turbine: float, failure_rate_type: str, floating: bool):
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
