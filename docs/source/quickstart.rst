Quick Start
===========

This project provides a framework for simulating offshore wind
operation and maintenance (O&M) costs, downtime, and availability
under uncertainty.

Prerequisites
-------------

Before using the package, install:

* `Python <https://www.python.org/downloads/>`_. 3.10 or newer
* `Git <https://git-scm.com/downloads>`_

Installation
------------

Clone the repository:

.. code-block:: bash

   git clone git@github.com:eliastche/om-pre-processing.git
   cd offshore-wind-om

Install the package in editable mode:

.. code-block:: bash

   pip install -e .

This installs the package and its dependencies while allowing
local code changes to be reflected immediately without reinstalling.

Running a Simulation
--------------------

Configure your scenario and execute the main run script see the example.ipynb notebook for a complete example.:

.. code-block:: python

   from offshore_om.run import run_master

   results = run_master(...)

