#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A script that dispatches all necessary routnes
and generates relevant outputs
"""

#
# Copyright (c) 2025 John Isaac Calderon

# Necessary modules for parsing and general analysis
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy

# OS-related hooks
import os
import sys

# Import sub-scripts
# - these scripts satisfy tasks 1, 2
from scripts import run_provided, task_2

# Run the provided script (task 1), then take
# ownership of the returned station data
print(" I: (run_all) Running `run_provided.main()`...")
s_data, s_times, s_lats, s_lons, s_dt = run_provided.main()

# Reset PyPlot to default settings
mpl.rcParams.update(mpl.rcParamsDefault)

# Run main routine for task 2, then take
# ownereship of the returned filtered data
# - don't show section data just yet
print(" I: (run_all) Running `task_2.main(...)`...")
filtered = task_2.main(1000, 100.0, s_data, False, True)
