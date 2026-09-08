"""
Failure and maintenance campaign data structures.

This module defines the objects used to represent component failures and
maintenance campaigns during the offshore wind O&M simulation.

Failures store information about individual component breakdown events,
while campaigns manage vessel mobilization, repair backlogs, spare-part
inventory, and weather-related delays.
"""

from dataclasses import dataclass
import numpy as np
from .economics import get_season_from_time
from .constants import HOURS_PER_YEAR

@dataclass
class Failure:
    """
    Representation of a component failure within a wind turbine.

    A failure corresponds to one component becoming unavailable at a
    specific point in time. The failure is subsequently assigned to a
    maintenance campaign and repaired by the simulation.

    Attributes
    ----------
    turbine_id : int
        Identifier of the affected turbine.
    component_id : int
        Identifier of the failed component within the turbine.
    component_name : str
        Human-readable component name.
    severity : str
        Failure severity category.
    failure_time_h : float
        Time of failure in simulation hours.
    """
    turbine_id: int
    component_id: int
    component_name: str
    severity: str
    failure_time_h: float

class Campaign:
    """
    Maintenance campaign performed by a repair vessel.

    A campaign represents the complete repair operation associated with a
    failure event, including vessel mobilization, weather delays,
    transportation, repair execution, spare-part management, and vessel
    return to port.

    Parameters
    ----------
    vessel : dict
        Vessel specification containing operational and cost parameters.
    start_time_h : float
        Campaign start time in simulation hours.
    parts_capacity : int
        Number of replacement parts that can be carried at one time.

    Attributes
    ----------
    vessel : dict
        Vessel assigned to the campaign.
    start_time_h : float
        Campaign start time in simulation hours.
    parts_capacity : int
        Maximum spare-part inventory.
    parts_remaining : int
        Number of spare parts currently available.
    failures : list of Failure
        Failures assigned to the campaign.
    vessel_at_farm : bool
        Whether the vessel is currently located at the wind farm.
    vessel_returned_to_port : bool
        Whether the vessel has completed its return voyage.
    crew_workers : int
        Number of workers assigned to the campaign.
    charter_start_h : float or None
        Time at which vessel chartering begins.
    charter_end_h : float or None
        Time at which vessel chartering ends.
    """
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
        """
        Calculate vessel arrival time at the wind farm.

        Returns
        -------
        float
            Arrival time in simulation hours.
        """
        return self.start_time_h + self.vessel["mobilisation_hours"]

    def weather_delay_hours(self, arrival_time_h: float, rng: np.random.Generator):
        """
        Calculate weather-related mobilisation delay.

        Delay duration depends on the season and a stochastic draw from
        the supplied random-number generator.

        Parameters
        ----------
        arrival_time_h : float
            Scheduled vessel arrival time in simulation hours.
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        float
            Additional weather delay in hours.
        """
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
        """
        Add a failure to the campaign backlog.

        Parameters
        ----------
        failure : Failure
            Failure to add to the campaign.
        """
        self.failures.append(failure)

    def has_backlog(self):
        """
        Check whether the campaign contains unresolved failures.

        Returns
        -------
        bool
            True if at least one failure remains in the backlog,
            otherwise False.
        """
        return len(self.failures) > 0

    def pop_next_failure(self):
        """
        Retrieve and remove the next failure in the campaign backlog.

        Returns
        -------
        Failure
            Next failure scheduled for repair.
        """

        return self.failures.pop(0)

    def use_part(self):
        """
        Consume one replacement part from inventory.

        Notes
        -----
        This method reduces the available spare-part count by one.
        """
        self.parts_remaining -= 1

    def needs_resupply(self):
        """
        Determine whether the campaign requires resupply.

        Returns
        -------
        bool
            True if no spare parts remain available.
        """
        return self.parts_remaining <= 0

    def resupply(self):
        """
        Replenish campaign spare-part inventory.

        Notes
        -----
        Inventory is restored to full capacity and the vessel is marked as
        being away from the wind farm, requiring a transit back before
        repairs can resume.
        """
        self.parts_remaining = self.parts_capacity
        self.vessel_at_farm = False
