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
from scripts import run_provided, task_2


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
    print(" I: (run_all) Running `run_provided.main()`...")
    s_data, s_times, s_lats, s_lons, s_dt, s_dists, figs_1 = run_provided.main(
        show_plots=False
    )

    # - make sure we get figures in return
    assert all(f is not None for f in figs_1), "Not all figures were returned"
    plt.show()

    # Reset PyPlot to default settings
    mpl.rcParams.update(mpl.rcParamsDefault)

    # Run main routine for task 2, then take
    # ownereship of the returned filtered data
    print(" I: (run_all) Running `task_2.main(...)`...")
    filtered, figs_2 = task_2.main(
        1000,
        100.0,
        s_data,
        s_times,
        s_dists,
        show_section=True,
        show_plots=False,
        fallback=False,
    )

    # - make sure we get filtered data and figures in return,
    # as we 1) have s_data, and 2) want to show the plots elsewhere
    assert filtered is not None, "Filtered data weren't returned"
    assert figs_2 is not None, "No figures were returned"

    print(" I: %d figures were generated" % (len(figs_1) + len(figs_2)))
    plt.show()


# PROPER IDIOM MUST BE USED
if __name__ == "__main__":
    main()
