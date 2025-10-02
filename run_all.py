#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A script that dispatches all necessary routnes
and generates relevant outputs
"""

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
from scripts import run_provided, task_2, task_3


def main():
    """
    Main routine

    This routine *must* exist in order for spawn-based
    multi-processing to work at all.

    Parameters
    ----------
    Does not accept any parameters

    Returns
    -------
    Does not return any values.
    """

    # Run the provided script (task 1), then take
    # ownership of the returned station data
    print(" I: (run_all) Running `run_provided.main()`...", file=sys.stderr)
    s_data, s_times, s_lats, s_lons, s_dt, s_dists, _ = run_provided.main(
        show_plots=True
    )

    # Reset PyPlot to default settings
    mpl.rcParams.update(mpl.rcParamsDefault)

    # Run main routine for task 2, then take
    # ownereship of the returned filtered data
    print(" I: (run_all) Running `task_2.main(...)`...", file=sys.stderr)
    filtered, figs_2 = task_2.main(
        1000,
        100.0,
        s_data,
        s_times,
        s_dists,
        show_section=True,
        show_plots=True,
        fallback=False,
    )

    # - make sure we get filtered data and figures in return,
    # as we 1) have s_data, and 2) want to show the plots elsewhere
    assert filtered is not None, "Filtered data weren't returned"

    # Reset PyPlot to default settings
    mpl.rcParams.update(mpl.rcParamsDefault)

    # Unpacked filtered data, then run main routine for task 3
    print(" I: (run_all) Running `task_3.main(...)`...", file=sys.stderr)

    fd1, fd2, fd3 = filtered
    task_3.main(
        s_dists, s_times, s_data, fd1, fd2, fd3, reuse=True, show_plots=True
    )


# PROPER IDIOM MUST BE USED
if __name__ == "__main__":
    main()
