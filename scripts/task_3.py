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
import matplotlib.figure
from scipy.stats import linregress

# Type hints
from typing import Optional, Tuple

# Timekeeping
import time

# OS-related hooks
import os
import sys

# Obtain current directory
cwd = os.path.abspath("")

# Set results path
result_path = os.path.join(cwd, "task_3a.npz")
tmp_path = os.path.join(cwd, "task_3a_tmp.npz")


# Method to satisfy task 3a
def task_3a(
    dists: np.ndarray,
    times: np.ndarray,
    d0: np.ndarray,
    fd1: np.ndarray,
    fd2: np.ndarray,
    fd3: np.ndarray,
    reuse: bool = True,
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
    reuse : bool = True
        Determines whether to re-use saved results

    Returns
    -------
    Does not return any values
    """

    print(" I: Executing task 3a")

    # Reuse results if requested
    if os.path.exists(result_path) and reuse:
        print(" I: (task_3a) Reusing old results")

        try:
            # - use Numpy as backend
            with np.load(result_path, allow_pickle=True) as f:
                result = f["result"]

            # - validate result
            assert len(result.shape) == 2, "Loaded result has invalid shape"
            assert result.shape[0] == 3, "Loaded result has invalid shape"

            # - return loaded result
            return result
        except Exception as e:
            print(f" W: (task_3a) Failed to load old results ({e})")

    # Prepare a selector list, which is
    # essentially an array of the distances,
    # tagged by their index
    dists_sorted = [(i, j) for i, j in enumerate(dists)]

    # - sort by distance (second entry in tuple)
    dists_sorted.sort(key=lambda x: x[1])

    # Set up result matrix
    # - row 0: station ID
    # - row 1: distance
    # - row 2: selected time
    result = [[], [], []]

    print(" W: (task_3a) Possible SIGSEGV on exit")
    print(" W: (task_3a) Triple-click and submit to end session gracefully")

    # Loop over the stations, sorted by distance
    for k, (i, j) in enumerate(dists_sorted):
        # Obtain figure and axes
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)

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
        try:
            # - poll from GUI
            x = fig.ginput(n=-1)

            # - break loop if 3 or more points are added,
            # as closing the window does NOT end the session
            # gracefully (leading to SIGSEGV)
            if len(x) > 2:
                break

            # - save only if there is exactly one point in the buffer
            if len(x) == 1:
                # - save ID, distance, selected time
                result[0].append(i)
                result[1].append(j)
                result[2].append(x[0][0])

        except Exception as e:
            print(f" W: `ginput` aborted ({e})")
        finally:
            plt.close(fig)

    # Convert result matrix to Numpy array
    result = np.array(result)

    # Save result matrix to file
    try:
        # - save to temporary file
        np.savez(tmp_path, result=result, allow_pickle=True)

        # - move to `.npz` file
        os.replace(tmp_path, result_path)
    except Exception as e:
        print(f" W: (task_3a) Failed to store results ({e})")

    # - return results anyways
    return result


# Method to satisfy task 3b
def task_3b(
    times: np.ndarray, dists: np.ndarray, show: bool = True
) -> matplotlib.figure.Figure:
    """
    A method that satisfies task 3b
    """

    # Perform linear regression on the provided data
    reg_res = linregress(times, dists)
    m, y0, rel = reg_res.slope, reg_res.intercept, reg_res.rvalue

    # Calculate celerity (in kilometers per hour)
    # - time is in days; convert to hours
    cel = m / 24
    cel_si = cel / 3.6  # - convert to meters per second

    print(f" I: (task_3b) Estimated celerity: {cel:.1f} km/h, {cel_si:.2f} m/s")
    print(f" I: (task_3b) R-squared value: {rel**2:.3f}")

    # Obtain regression values
    reg_x = np.linspace(times.min(), times.max(), 1000)
    reg_y = m * reg_x + y0

    # Instantiate figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Set up locator and formatter
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    # Plot data and set axis labels
    ax.plot(times, dists, ".", label="Raw times")
    ax.plot(reg_x, reg_y, "--", label=f"Fitted curve (cel.: {cel:.1f} km/h)")

    ax.set_xlabel("Arrival time (UTC)")
    ax.set_ylabel("Station distance [km]")

    # - assign formatter and locator
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(locator)

    # Set figure and subplot titles
    fig.suptitle("Task 3b")
    ax.set_title("Station distance versus arrival time")

    # Enable grid and legend
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()

    # Show if requested
    if show:
        plt.show()
        return None
    else:
        return fig


# Main routine for task 3
def main(
    dists: np.ndarray,
    times: np.ndarray,
    d0: np.ndarray,
    fd1: np.ndarray,
    fd2: np.ndarray,
    fd3: np.ndarray,
    reuse: bool = True,
    show_plots: bool = True,
) -> Optional[matplotlib.figure.Figure]:
    """
    Main routine for task 3
    """

    # Obtain picked times
    res = task_3a(dists, times, d0, fd1, fd2, fd3, reuse)

    # Plot picked times
    fig = task_3b(res[2], res[1], show=show_plots)

    # Return figure
    return fig


# Disallow direct execution
if __name__ == "__main__":
    print(" E: Direct execution not allowed")
    print("Please use `run_all.py' instead")
    sys.exit(1)
