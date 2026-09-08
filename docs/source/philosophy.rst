Philosophy
==========

Purpose
-------

The Offshore O&M Risk Model was developed to quantify the impact of
uncertainty in offshore wind operation and maintenance (O&M)
activities. Offshore wind farms are exposed to stochastic failures,
weather-dependent accessibility constraints, and complex maintenance
campaigns that can significantly affect costs and availability.

Traditional analyses often rely on deterministic assumptions and
average values, which may underestimate the financial and operational
risks associated with offshore wind assets. This framework aims to
provide a more realistic representation by explicitly modelling
uncertainty and producing probability distributions of outcomes
rather than single-point estimates.

Goals
-----

The primary goals of the model are to:

* Simulate offshore wind farm operation and maintenance activities.
* Quantify uncertainty in costs, downtime, and availability.
* Identify operational and economic risk drivers.
* Support scenario analysis and sensitivity studies.
* Generate inputs for downstream energy system and techno-economic
  analyses.

Design Philosophy
-----------------

The framework builds upon the maintenance cost modelling approach
presented by Mikindani et al. :cite:p:`mikindaniFinancialRisksWind2025` and extends it to offshore wind
applications. While the original work focused on maintenance cost
assessment, this framework incorporates offshore wind-specific
characteristics such as weather accessibility constraints,
campaign-based maintenance strategies, and site-specific logistics.

The maintenance modelling philosophy draws on concepts commonly used
throughout the offshore wind O&M literature, including the work of
Carroll et al. :cite:p:`carrollFailureRateRepair2016`, Dalgic et al. :cite:p:`dalgicInvestigationOptimumJackup2015`, Donnelly et al. :cite:p:`donnellyOperationMaintenanceCost2024`, and Dighe et al. :cite:p:`digheOffshoreWindFarm2026`.
The objective is not to introduce entirely new maintenance concepts,
but rather to combine established methodologies into a flexible and
transparent framework suitable for uncertainty and risk analysis.

Defensible Data and Assumptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A central design objective is that input assumptions should be
traceable and defensible. Failure rates, repair durations, campaign
strategies, accessibility constraints, and economic parameters are
explicitly defined and can be linked to literature sources or project-
specific assumptions. This ensures that scenario results can be
understood, challenged, and refined as new information becomes
available.

Modularity and Extensibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The framework is intentionally modular. Components, maintenance
strategies, economic assumptions, and offshore wind areas are
represented independently, allowing users to modify individual
elements without affecting the broader simulation structure.

This modularity simplifies sensitivity analyses and scenario studies,
as assumptions can be adjusted systematically while maintaining a
consistent simulation workflow.

Campaign-Based Maintenance
^^^^^^^^^^^^^^^^^^^^^^^^^^

Maintenance activities are organized through campaign-based logic,
reflecting how major maintenance operations are commonly planned in
offshore wind projects. This approach allows the framework to capture
the interaction between maintenance scheduling, vessel mobilization,
accessibility windows, and weather-related delays while remaining
computationally efficient for large Monte Carlo studies.

Support for Multiple Offshore Wind Areas
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The framework is designed to easily accommodate new offshore wind
areas with unique environmental and economic characteristics. Site-
specific assumptions can therefore be introduced without modifying the
core simulation engine, enabling comparative analyses across regions,
projects, and future development scenarios.

Scope
-----

The primary purpose of the framework is to quantify how operational
risk influences offshore wind costs and availability assessing the
economic consequences of uncertainty in energy models.

The framework is therefore intended as a stochastic decision-support framework
for understanding the range of possible outcomes and evaluating the
risks associated with offshore wind operation and maintenance.

References
----------

.. bibliography::