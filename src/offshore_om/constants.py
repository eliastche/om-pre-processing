"""
Default model constants and reference datasets.

This module contains the default assumptions used throughout the
offshore wind O&M simulation framework. The constants defined here
represent reference values for component reliability, maintenance
requirements, vessel characteristics, and offshore wind region
configurations.

Contents
--------
component_data
    Default component reliability and repair database. Contains
    Weibull failure parameters, severity distributions, repair
    durations, material costs, technician requirements, and
    failure-rate assumptions for major turbine components.

vessel_data
    Default vessel database used for maintenance logistics.
    Includes vessel day rates, mobilisation costs, mobilisation
    durations, and transit speeds for different maintenance
    severities.

WindRegions
    Collection of predefined Norwegian offshore wind regions used
    in simulation studies. Each region contains assumptions for
    capacity factor, distance to shore, and technology type
    (fixed-bottom or floating).

Model Constants
---------------
HOURS_PER_YEAR
    Number of hours per year (8760).

kNOKEUR
    Currency conversion factor from EUR to kNOK.

kNOKGBP
    Currency conversion factor from GBP to kNOK.

ZERO_TOLERANCE
    Numerical tolerance used when comparing floating-point values.

Data Sources
------------
The default component reliability and maintenance assumptions are
derived from published offshore wind O&M literature, including
studies by Mikindani et al., Carroll et al., Dalgic et al.,
Donnelly et al., Hendriks et al., and Jenkins et al. The values
represent engineering assumptions intended for scenario analysis
rather than precise site-specific forecasts.

The vessel and regional datasets are based on assumptions used in
the IFE offshore wind maintenance modelling framework and can be
replaced or extended for alternative case studies.

Notes
-----
This module contains default reference data only. The associated
data structures are defined in ``components.py``. Users are
encouraged to modify or replace these assumptions through scenario
configuration when conducting project-specific analyses.
"""

from offshore_om.components import WindRegion

HOURS_PER_YEAR = 8760
kNOKEUR = 10.39 / 1000
kNOKGBP = 11.385 / 1000
ZERO_TOLERANCE = 1e-2


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
