#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A script that dispatches all necessary routnes
and generates relevant outputs
"""

#
# Copyright (c) 2025 John Isaac Calderon

# Necessary modules for parsing and general analysis
import matplotlib.pyplot as plt
import numpy as np
import scipy

# OS-related hooks
import os
import sys

# Modules to speed up calculations
from numba import njit
import multiprocessing

# Import sub-scripts
# - these scripts satisfy tasks 1, 2
from scripts import run_provided, task_2

# Run the provided script (task 1), then take
# ownership of the returned station data
s_data, s_times, s_lats, s_lons, s_dt = run_provided.main()

# Run main routine for task 2, then take
# ownereship of the returned filtered data
# - don't show section data just yet
filtered = task_2.main(1000, 100.0, s_data, False)
