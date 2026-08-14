from dataclasses import dataclass

@dataclass
class CostEvent:
    time_years: float
    component_name: str

    downtime_hours: float
    lost_mwh: float

    material_cost: float
    labour_cost: float
    transport_cost: float

    downtime_start_years: float | None = None
    downtime_end_years: float | None = None

    @property
    def direct_cost(self):
        return self.material_cost + self.labour_cost + self.transport_cost

class CostTracker:
    def __init__(self):
        self.events = []

    def add_event(
        self,
        time_years: float,
        component_name: str,
        downtime_hours: float,
        lost_mwh: float,
        material_cost: float = 0.0,
        labour_cost: float = 0.0,
        transport_cost: float = 0.0,
        downtime_start_years: float | None = None,
        downtime_end_years: float | None = None,
    ):
        self.events.append(
            CostEvent(
                time_years=time_years,
                component_name=component_name,
                downtime_hours=downtime_hours,
                lost_mwh=lost_mwh,
                material_cost=material_cost,
                labour_cost=labour_cost,
                transport_cost=transport_cost,
                downtime_start_years=downtime_start_years,
                downtime_end_years=downtime_end_years,
            )
        )

    @property
    def total_downtime_hours(self):
        return sum(event.downtime_hours for event in self.events)

    @property
    def total_lost_mwh(self):
        return sum(event.lost_mwh for event in self.events)

    @property
    def total_material_cost(self):
        return sum(event.material_cost for event in self.events)

    @property
    def total_labour_cost(self):
        return sum(event.labour_cost for event in self.events)

    @property
    def total_transport_cost(self):
        return sum(event.transport_cost for event in self.events)

    @property
    def total_direct_cost(self):
        return sum(event.direct_cost for event in self.events)