from dataclasses import dataclass
import numpy as np
from .economics import get_season_from_time
from .constants import HOURS_PER_YEAR

@dataclass
class Failure:
    turbine_id: int
    component_id: int
    component_name: str
    severity: str
    failure_time_h: float

class Campaign:
    def __init__(self, vessel: dict, start_time_h: float, parts_capacity: int):
        self.vessel = vessel
        self.start_time_h = start_time_h
        self.parts_capacity = parts_capacity
        self.parts_remaining = parts_capacity
        self.failures = []
        self.vessel_at_farm = False
        self.vessel_returned_to_port = False
        self.crew_workers = 0
        self.charter_start_h = None
        self.charter_end_h = None

    @property
    def arrival_time_h(self):
        return self.start_time_h + self.vessel["mobilisation_hours"]

    def weather_delay_hours(self, arrival_time_h: float, rng: np.random.Generator):
        season = get_season_from_time(arrival_time_h / HOURS_PER_YEAR)
        chance = rng.uniform(0, 1)
        delay_hours = 0
        if season == "Spring" and chance > 0.6:
            delay_hours = 120
        elif season == "Summer" and chance > 0.8:
            delay_hours = 48
        elif season == "Fall" and chance > 0.5:
            delay_hours = 72
        elif season == "Winter" and chance > 0.3:
            delay_hours = 168
        return delay_hours

    def add_failure(self, failure: Failure):
        self.failures.append(failure)

    def has_backlog(self):
        return len(self.failures) > 0

    def pop_next_failure(self):
        return self.failures.pop(0)

    def use_part(self):
        self.parts_remaining -= 1

    def needs_resupply(self):
        return self.parts_remaining <= 0

    def resupply(self):
        self.parts_remaining = self.parts_capacity
        self.vessel_at_farm = False
