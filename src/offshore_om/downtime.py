"""
Cost and downtime tracking utilities.

This module defines the data structures used to record maintenance events,
downtime, production losses, and economic impacts throughout an offshore
wind O&M simulation.

Individual maintenance events are represented by :class:`CostEvent`,
while :class:`CostTracker` provides aggregation of operational and
economic metrics across the simulation horizon.
"""

from dataclasses import dataclass

@dataclass
class CostEvent:
    """
    Record of a maintenance-related event.

    A cost event captures the timing and consequences of a repair,
    campaign, or other operational activity. Events may contribute to
    downtime, lost energy production, labour costs, transport costs,
    and material costs.

    Attributes
    ----------
    time_years : float
        Time at which the event is recorded, expressed in simulation years.
    component_name : str
        Name of the affected component or activity.
    downtime_hours : float
        Duration of turbine downtime caused by the event.
    lost_mwh : float
        Energy production lost due to downtime.
    material_cost : float
        Material expenditure associated with the event.
    labour_cost : float
        Labour expenditure associated with the event.
    transport_cost : float
        Vessel or transportation expenditure associated with the event.
    downtime_start_years : float or None, optional
        Time at which downtime began.
    downtime_end_years : float or None, optional
        Time at which downtime ended.
    """
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
        """
        Calculate the direct cost associated with the event.

        Returns
        -------
        float
            Sum of material, labour, and transport costs.
        """
        return self.material_cost + self.labour_cost + self.transport_cost

class CostTracker:
    """
    Collect and aggregate simulation cost events.

    The tracker stores all maintenance-related events occurring during a
    simulation and provides aggregate statistics for downtime, lost
    production, and direct O&M costs.

    Attributes
    ----------
    events : list of CostEvent
        Recorded simulation events.
    """
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

        """
        Record a simulation event.

        Parameters
        ----------
        time_years : float
            Event occurrence time in simulation years.
        component_name : str
            Name of the affected component or activity.
        downtime_hours : float
            Downtime caused by the event.
        lost_mwh : float
            Lost energy production associated with the event.
        material_cost : float, optional
            Material expenditure. The default is 0.0.
        labour_cost : float, optional
            Labour expenditure. The default is 0.0.
        transport_cost : float, optional
            Transport expenditure. The default is 0.0.
        downtime_start_years : float or None, optional
            Downtime start time in simulation years.
        downtime_end_years : float or None, optional
            Downtime end time in simulation years.

        Notes
        -----
        A new :class:`CostEvent` is created and appended to the internal
        event list.
        """

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
        """
        Total downtime accumulated across all events.

        Returns
        -------
        float
            Downtime in hours.
        """
        return sum(event.downtime_hours for event in self.events)

    @property
    def total_lost_mwh(self):
        """
        Total lost energy production.

        Returns
        -------
        float
            Lost energy production in MWh.
        """
        return sum(event.lost_mwh for event in self.events)

    @property
    def total_material_cost(self):
        """
        Total material costs.

        Returns
        -------
        float
            Material costs summed across all events.
        """
        return sum(event.material_cost for event in self.events)

    @property
    def total_labour_cost(self):
        """
        Total labour costs.

        Returns
        -------
        float
            Labour costs summed across all events.
        """
        return sum(event.labour_cost for event in self.events)

    @property
    def total_transport_cost(self):
        """
        Total transport costs.

        Returns
        -------
        float
            Transportation costs summed across all events.
        """
        return sum(event.transport_cost for event in self.events)

    @property
    def total_direct_cost(self):
        """
        Total direct costs.

        Returns
        -------
        float
            Total direct costs.
        """
        return sum(event.direct_cost for event in self.events)