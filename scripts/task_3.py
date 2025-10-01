#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A script that satisfies task 3 (Estimating celerity)
"""

# Copyright (c) 2025 John Isaac Calderon

# Import modules for general analysis
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Type hints
from typing import Optional, Tuple

# Timekeeping
import time

# OS-related hooks
import os
import sys

# Obtain current directory
cwd = os.path.abspath("")


# Method to satisfy task 3a
def task_3a(
    dists: np.ndarray,
    times: np.ndarray,
    d0: np.ndarray,
    fd1: np.ndarray,
    fd2: np.ndarray,
    fd3: np.ndarray,
):
    """
    A method that satisfies task 3a

    Parameters
    ----------
    dists : np.ndarray
        (S,) array containing the distances
        between each station and the event
    times : np.ndarray
        (S, T) array containing the wall time
        at each station
    d0 : np.ndarray
        (S, T) array containing raw measurements
        from each station
    fd1, fd2, fd3 : np.ndarray
        (S, T) arrays containing filtered measurements

        The signals are expected to be derived from the
        convolution between `d0` and
        1. a low-pass filter,
        2. a band-pass filter, and
        3. a high-pass filter

    Returns
    -------
    Does not return any values
    """

    print(" I: Executing task 3a")

    # Prepare a selector list, which is
    # essentially an array of the distances,
    # tagged by their index
    dists_sorted = [(i, j) for i, j in enumerate(dists)]

    # - sort by distance (second entry in tuple)
    dists_sorted.sort(key=lambda x: x[1])

    # Loop over the stations, sorted by distance
    for k, (i, j) in enumerate(dists_sorted):
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

        # Set figure size
        fig.set_size_inches(14, 8, forward=True)

        # Set up locator and formatter
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)

        # Initialize each subplot
        titles = ["Raw", "Low-pass", "Band-pass", "High-pass"]

        for a, t in zip(axes.flatten(), titles):
            # - set axis formatter and locator
            a.xaxis.set_major_formatter(formatter)
            a.xaxis.set_major_locator(locator)

            # - set axis labels
            a.set_xlabel("UTC Time")
            a.set_ylabel("Signal magnitude [a.u.]")

            # - set grid
            a.grid()

            # - set subplot title
            a.set_title(t)

        # Plot time series
        axes[0, 0].plot(times[i], d0[i])
        axes[0, 1].plot(times[i], fd1[i])
        axes[1, 0].plot(times[i], fd2[i])
        axes[1, 1].plot(times[i], fd3[i])

        # Set figure title
        fig.suptitle(f"Station #{i} ({k+1}/{len(dists)}); dist.: {j:.3f} km")

        # Show figure
        plt.show()


# Main routine for task 3
def main(
    dists: np.ndarray,
    times: np.ndarray,
    d0: np.ndarray,
    fd1: np.ndarray,
    fd2: np.ndarray,
    fd3: np.ndarray,
):
    """
    Main routine for task 3
    """

    task_3a(dists, times, d0, fd1, fd2, fd3)


# Disallow direct execution
if __name__ == "__main__":
    print(" E: Direct execution not allowed")
    print("Please use `run_all.py' instead")
    sys.exit(1)
