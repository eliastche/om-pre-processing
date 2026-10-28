Data Basis
==========

The offshore wind O&M model combines technical, operational,
environmental, and economic input data. These data define the
failure behaviour, maintenance requirements, accessibility
constraints, and cost structure used during simulation.

Component and Failure Data
--------------------------

Component-specific failure and repair parameters are defined
in the ``components`` module. For each component type, the
model requires Weibull failure parameters, repair durations,
crew requirements, spare-part costs, and severity
distributions.

The default values are derived from published offshore wind
O&M literature, including the shape parameter estimates
of :cite:p:`mikindaniFinancialRisksWind2025`. and subsequent offshore wind maintenance
studies by :cite:p:`carrollEconomicAnalysisWind2025`, :cite:p:`dalgicEconomicAnalysisWind2025`, :cite:p:`donnellyEconomicAnalysisWind2025` and others. These values
represent engineering assumptions and can be modified for
alternative studies.

Vessel Data
-----------

Vessel characteristics are defined through a collection of
vessel types used during maintenance operations. Parameters
include transit speed, mobilisation duration, mobilisation
cost and daily charter rate.

These inputs determine both maintenance costs and operational
downtime associated with repair activities.

Site Characteristics
--------------------

Regional offshore wind site characteristics are represented
through the ``WindRegion`` configuration. These parameters
include distance to shore, capacity factor and technology
type (fixed-bottom or floating).

The regional definitions determine energy production,
maintenance logistics and weather-related accessibility.

Weather Data
------------

Weather accessibility is evaluated using historical weather
time series. Vessel operating limits are applied to determine
whether maintenance operations can be executed at a given
time.

Economic Data
-------------

Economic assumptions include electricity prices, vessel
rates, labour costs and discount rates. These parameters are
used when calculating maintenance expenditures and financial
risk metrics.

Limitations
-----------

The reliability of simulation outputs depends directly on the
quality of the underlying assumptions. Failure-rate estimates
and repair characteristics remain uncertain, particularly for
large-scale floating offshore wind developments. Simulation
results should therefore be interpreted as scenario-based
estimates rather than precise forecasts.