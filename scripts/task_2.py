#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A script that satisfies task 2 (Data filtering)
"""

# Copyright (c) 2025 John Isaac Calderon

# Import FIR filters
from .fir import h1, h2, h3, LEN_H1, LEN_H2, LEN_H3

# Import custom definitions
from . import is_older

# Import modules for general analysis
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from scipy.signal import fftconvolve
from scipy.fft import rfft, irfft

# Systems acceleration
from numba import njit, prange

# Type hints
from typing import Optional, Tuple

# Timekeeping
import time

# OS-related hooks
import os

# Obtain current directory
cwd = os.path.abspath("")

# Obtain logical cores count
logical_cores = os.cpu_count()

# - calculate number of free cores
reserved_cores = 2
free_cores = logical_cores - reserved_cores


# Performs linear convolution between `x` and `h`
# - use fast methods and strict type discipline
@njit(parallel=True, fastmath=True)
def convolve(x: np.ndarray, h: np.ndarray, ylen_choice: int) -> np.ndarray:
    """
    Performs linear convolution between `x` and `h`

    Implements linear convolution using a "schoolbook" kernel,
    which is slow for most purposes. Use `np.convolve`,
    `scipy.signal.fftconvolve`, or similar algorithms instead.

    Parameters
    ----------
    x : np.ndarray
        An (M,)-shaped array
    h : np.ndarray
        An (N,)-shaped array
    ylen_choice : int
        Determines convolution mode

        If `ylen_choice == 0`, then the output will have
        the shape (M,), whereas if `ylen_choice == 1`,
        then the output will have the shape (M + N - 1).

    Returns
    -------
    y : np.ndarray
        Linear convolution of `x` and `h`
    """

    # Even if the task doesn't demand it, we'll try
    # to speed up the convolution process by using
    # whatever trick we can come up with

    # Calculate output length
    M, N = len(x), len(h)
    ylen = M + N - 1

    # Use contiguous arrays
    # - pre-reverse `h` to avoid backward strides
    x = np.ascontiguousarray(x)
    h_rev = np.ascontiguousarray(h[::-1])

    # Set up output vector
    y = np.zeros(ylen)

    # Convolve for each item in the vector
    # - perform serial computes
    # - calculate the full series
    for n in prange(ylen):
        # - accumulator
        acc = 0

        # - calculate convolution
        for k in prange(M):
            # - calculate forward-striding index
            j = (N - 1 - n) + k

            if 0 <= j < N:
                # - kernel
                acc += x[k] * h_rev[j]

        # - store accumulator value
        y[n] = acc

    # - return output vector
    if ylen_choice:
        # - return full vector
        return y
    else:
        # - return central part with size `M`
        start = (N - 1) // 2
        return y[start : start + M]


# Helper function for `dtft`
# - use separate helper function with strict type discipline
@njit(parallel=True, fastmath=True)
def __dtft(h: np.ndarray, N: int, fs: float, freqs: np.ndarray) -> np.ndarray:
    """
    Helper function for `dtft`

    Calculates the DTFT of the impulse response `h` using
    a direct (schoolbook) method.

    Parameters
    ----------
    h : np.ndarray
        Impulse response to be analyzed
    N : int
        Number of points on the unit circle
    fs : float
        Sampling frequency (in hertz)
    freqs : np.ndarray
        Array containing frequencies

    Returns
    -------
    H : np.ndarray
        Complex-valued DTFT of the provided impulse response
    """

    # Set up output vector
    H = np.zeros_like(freqs, dtype=np.complex128)

    # Obtain indices
    n = np.arange(len(h))

    # Calculate DTFT using an outer loop
    # - use `np.sum` internally
    for k in prange(N):
        H[k] = np.sum(h * np.exp(-2j * np.pi * freqs[k] * n / fs))

    return H


# Performs DTFT on the provided impulse response
# - uses helper function `__dtft`
def dtft(h: np.ndarray, N: int, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs direct (schoolbook) DTFT on the provided impulse response

    This implementation is for pedagogical purposes;
    dedicated FFT algorithms should be used instead.

    Parameters
    ----------
    h : np.ndarray
        Impulse response to be analyzed
    N : int
        Number of points on the unit circle
    fs : float
        Sampling frequency (in hertz)

    Returns
    -------
    H : np.ndarray
        Complex-valued DTFT of the provided impulse response
    freqs : np.ndarray
        Frequencies [0, fs) in hertz
    """

    # Calculate frequencies
    freqs = np.linspace(0, fs, N, endpoint=False)

    # Calculate DTFT (see `__dtft` above)
    H = __dtft(h, N, fs, freqs)

    # Return DTFT and frequencies
    return H, freqs


# Convolve given signal matrix with FIR filter `h1`
def __convolve_h1(m: np.ndarray) -> np.ndarray:
    y = np.apply_along_axis(lambda x: fftconvolve(x, h1, mode="same"), 1, m)
    return y


# Convolve given signal matrix with FIR filter `h2`
def __convolve_h2(m: np.ndarray) -> np.ndarray:
    y = np.apply_along_axis(lambda x: fftconvolve(x, h2, mode="same"), 1, m)
    return y


# Convolve given signal matrix with FIR filter `h3`
def __convolve_h3(m: np.ndarray) -> np.ndarray:
    y = np.apply_along_axis(lambda x: fftconvolve(x, h3, mode="same"), 1, m)
    return y


def task_2a(show: bool = True) -> Optional[matplotlib.figure.Figure]:
    """
    A method that satisfies task 2a

    Parameters
    ----------
    show : bool = True
        Determines whether the generated figure should be shown

        If `show == False`, then the function returns
        the figure that would be shown.

    Returns
    -------
    fig : Optional[matplotlib.figure.Figure]
        Figure with the relevant axes
    """

    print(" I: Executing task 2a...")

    # Set up a range based on the smallest length
    least = min([LEN_H1, LEN_H2, LEN_H3])
    n = np.arange(0, least)

    # Set up figure and axes
    fig, ax = plt.subplots(1, 1)

    # Plot the impulse responses
    ax.plot(n, h1[n], "--.", label="$h_1$")
    ax.plot(n, h2[n], "--.", label="$h_2$")
    ax.plot(n, h3[n], "--.", label="$h_3$")

    # - set up grid and legend
    ax.grid()
    ax.legend()

    # - use proper title and axis labels
    fig.suptitle("Task 2a")
    ax.set_title("Impulse response of FIR filters")
    ax.set_xlabel("Input $n$ [1]")
    ax.set_ylabel("Impulse response $h[n]$ [1]")

    # - show figure
    if show:
        plt.show()
        return None
    else:
        return fig


def task_2b(show: bool = True) -> Optional[matplotlib.figure.Figure]:
    """
    A method that satisfies task 2b

    Parameters
    ----------
    show : bool = True
        Determines whether the generated figure should be shown

        If `show == False`, then the function returns
        the figure that would be shown.

    Returns
    -------
    fig : Optional[matplotlib.figure.Figure]
        Figure with the relevant axes
    """

    print(" I: Executing task 2b...")

    # Set up toy signal and impulse response
    x = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0])  # - rectangular pulse
    h = np.array([3, 2, 1, 0])  # - triangular filter

    # Calculate same and full convolutions
    c_same = convolve(x, h, 0)
    c_full = convolve(x, h, 1)
    ref_same = np.convolve(x, h, "same")
    ref_full = np.convolve(x, h, "full")

    # Set up index arrays
    n_same = np.arange(len(x))
    n_full = np.arange(len(x) + len(h) - 1)

    # Set up figure and axes
    fig, ax = plt.subplots(1, 1)

    # Plot calculated and reference values
    ax.plot(n_same, c_same, "-", label="same", color="yellow")
    ax.plot(n_same, ref_same, "--", label="same, reference", color="magenta")

    ax.plot(n_full, c_full, "-", label="full", color="cyan")
    ax.plot(n_full, ref_full, "--", label="full, reference", color="red")

    # Set up grid and legend
    ax.grid()
    ax.legend()

    # Set up title and axis labels
    fig.suptitle("Task 2b")
    ax.set_title("Comparison between hand-rolled and external functions")
    ax.set_xlabel("Discrete time $n$ [1]")
    ax.set_ylabel("Convolution $x * h$ [1]")

    # Show figure
    if show:
        plt.show()
        return None
    else:
        return fig


def task_2c():
    """
    A dummy method for task 2c

    Parameters
    ----------
    Does not accept any parameters

    Returns
    -------
    Does not return any values
    """
    print(" W: Task 2c is an implementation task")
    pass


def task_2d(
    n_points: int = 1000, fs: float = 100.0, show: bool = True
) -> Optional[matplotlib.figure.Figure]:
    """
    A method that satisfies task 2d

    Parameters
    ----------
    n_point : int = 1000
        Number of points on the unit circle
    fs : float = 100.0
        Sampling frequency (in hertz)
    show : bool = True
        Determines whether the generated figure should be shown

        If `show == False`, then the function returns
        the figure that would be shown.

    Returns
    -------
    fig : Optional[matplotlib.figure.Figure]
        Figure with the relevant axes
    """

    print(f" I: Executing task 2d (n_points = {n_points}, fs(.3f) = {fs:.3f} Hz)")

    # Calculate the DTFT of the impulse responses
    # - also receive the frequencies array
    H_1, freqs = dtft(h1, n_points, fs)
    H_2, _ = dtft(h2, n_points, fs)
    H_3, _ = dtft(h3, n_points, fs)

    # Calculate the magnitudes
    H_1_mag, H_2_mag, H_3_mag = np.abs(H_1), np.abs(H_2), np.abs(H_3)

    # Obtain only the first half of the spectrum
    # - include Nyquist frequency whenever possible
    freqs = freqs[: n_points // 2 + 1]
    H_1_mag = H_1_mag[: n_points // 2 + 1]
    H_2_mag = H_2_mag[: n_points // 2 + 1]
    H_3_mag = H_3_mag[: n_points // 2 + 1]

    # Set up figure and axes
    fig, ax = plt.subplots(1, 1)

    # Plot the magnitudes
    # - plot only the first half
    ax.plot(freqs, H_1_mag, "-", label="response $h_1$")
    ax.plot(freqs, H_2_mag, "--", label="response $h_2$")
    ax.plot(freqs, H_3_mag, "--", label="response $h_3$")

    # Set up grid and legend
    ax.grid()
    ax.legend()

    # Set up title and axis labels
    fig.suptitle("Task 2d")
    ax.set_title("Frequency response of FIR filters")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude $|H|$ [1]")

    # Show plot
    if show:
        plt.show()
        return None
    else:
        return fig


def task_2e():
    """
    A dummy method for task 2e

    Parameters
    ----------
    Does not accept any parameters

    Returns
    -------
    Does not return any values
    """

    print(" W: Task 2e is a qualitative task")
    pass


def task_2f(s_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    A method that satisfies task 2f

    Parameters
    ----------
    s_data : np.ndarray
        Array containing station data, with shape (S,T), where
        `S` is the number of stations, and `T` is the number of
        samples (separated by time)

    Returns
    -------
    res_1, res_2, res_3: np.ndarray
        Arrays with the shape (S,T), where `S` is the number
        of stations, and `T` is the number of samples
        (separated by time)
    """

    print(" I: Executing task 2f")

    t0 = time.time()

    master_cache = os.path.join(cwd, "unpacked.h5")
    slave_cache = os.path.join(cwd, "task_2f.npz")
    tmp_cache = os.path.join(cwd, "task_2f_tmp.npz")

    if is_older(master_cache, slave_cache):
        # Everything is as expected - load from cache
        print(" I: (task_2f) Reading from cache...")

        try:
            t0 = time.time()

            with np.load(slave_cache) as f:
                res_1 = f["res_1"]
                res_2 = f["res_2"]
                res_3 = f["res_3"]

            t1 = time.time()

            # Calculate size and transfer rate
            f_size = 3 * np.prod(res_1.shape) * 8
            rate = f_size / (t1 - t0) / (2**30)

            print(f" I: (read took {t1-t0:.3f} seconds; eff. rate: {rate:.3f} GiB/s)")

            return res_1, res_2, res_3
        except Exception as e:
            print(f" W: (task_2f) Failed to load from cache ({e})")
    # - fall-through

    # Perform manual FFT-based convolution
    # - this approach is supposedly faster, and saturates
    # memory lines earlier, though this would mean that
    # execution time is strongly dependent on the number
    # of RAM lanes, and the DRAM refresh rate

    N = s_data.shape[1]  # - obtain number of data points
    filters = [h1, h2, h3]  # - line up filters
    X = rfft(s_data, n=N, axis=1, workers=free_cores)  # - calculate FFT 2-tensor

    # - pad each filter to N, then calculate FFT
    H = np.stack([rfft(h, n=N, workers=free_cores) for h in filters])

    # Calculate resulting 3-tensor, which is the
    # convolution of `s_data` and all 3 kernels
    Y = irfft(H[:, None, :] * X[None, :, :], n=N, axis=2, workers=free_cores)

    # - split result by kernel (on planes 0, 1, 2)
    res_1, res_2, res_3 = Y[0], Y[1], Y[2]

    t1 = time.time()

    print(" I: (operation took %.3f seconds)" % (t1 - t0))

    # Build cache atomically
    try:
        # - save to temporary file
        np.savez(tmp_cache, res_1=res_1, res_2=res_2, res_3=res_3, allow_picle=True)

        # - move to `.npz` file
        os.replace(tmp_cache, slave_cache)
    except Exception as e:
        print(f" W: (task_2f) Failed to build cache ({e})")

    # Return filtered data
    return res_1, res_2, res_3


def task_2g(res: Optional[np.ndarray], show: bool = True):
    pass


def main(
    n_points: int = 1000,
    fs: float = 100.0,
    s_data: Optional[np.ndarray] = None,
    show_section: bool = False,
    show_plots: bool = True,
) -> Tuple[
    Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    Optional[Tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]],
]:
    """
    Main method for task 2

    Parameters
    ----------
    n_points : int = 1000
        Number of points on the unit circle
        (relevant for task 2d)
    fs : float = 100.0
        Sampling frequency (in hertz)
        (relevant for task 2d)
    s_data : Optional[np.ndarray] = None
        Data samples from stations
    show_section : bool = False
        Determines whether to show a section plot
    show_plots : bool = True
        Determines whether to show the plots in-place

        If `show_plots == False`, figures that would be
        shown will be returned instead.

    Returns
    -------
    (res_1, res_2, res_3) | None : Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        Filtered station data
    (fig_2a, fig_2b, fig_2d): Optional[Tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]]
        Figures for task 2b and 2d
    """

    # Set up temporary arrays and
    # warm up functions
    __r1 = np.random.rand(100)
    __r2 = np.random.rand(100)
    _ = convolve(__r1, __r2, 0)
    _ = convolve(__r2, __r1, 1)
    _ = dtft(__r1, 100, 100)
    _ = dtft(__r2, 100, 100)

    # - delete temporary arrays
    del __r1, __r2

    # Run the tasks in order
    fig_2a = task_2a(show=show_plots)
    fig_2b = task_2b(show=show_plots)
    task_2c()
    fig_2d = task_2d(n_points, fs, show=show_plots)
    task_2e()

    # - collect figures
    figs = (fig_2a, fig_2b, fig_2d) if not show_plots else None

    # - do not proceed if station data is not provided
    if s_data is None:
        return None, figs
    # - (fall-through)

    # Run tasks 2f and 2g
    res_1, res_2, res_3 = task_2f(s_data)

    if show_section:
        # - show section plot only if requested
        task_2g(None)
    elif show_plots:
        # - show filtered data for *one* station
        example_1 = res_1[0]
        example_2 = res_2[0]
        example_3 = res_3[0]

        # - generate time array in arbitrary units
        t = np.arange(len(example_1))

        # - plot filtered data
        plt.plot(t, example_1, label="$D_0 * h_1$")
        plt.plot(t, example_2, label="$D_0 * h_2$")
        plt.plot(t, example_3, label="$D_0 * h_3$")

        # - show grid and legend
        plt.grid()
        plt.legend()

        # - set up title and axis labels
        plt.title("Filtered station data")
        plt.xlabel("Generalized time [a.u.]")
        plt.ylabel("Station signal [a.u.]")

        # - show figure
        plt.show()

    # Return filtered data
    return (res_1, res_2, res_3), figs


# In case we get imported:
# set up temporary arrays
# and warm up functions
__r1 = np.random.rand(100)
__r2 = np.random.rand(100)
_ = convolve(__r1, __r2, 0)
_ = convolve(__r2, __r1, 1)
_ = dtft(__r1, 100, 100)
_ = dtft(__r2, 100, 100)

# - delete temporary arrays
del __r1, __r2
