#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A script that satisfies task 2 (Data filtering)
"""

# Copyright (c) 2025 John Isaac Calderon

# Import FIR filters
from .fir import h1, h2, h3, LEN_H1, LEN_H2, LEN_H3

# Import modules for general analysis
import numpy as np
import matplotlib.pyplot as plt

# Systems acceleration
from numba import njit, prange

# Type hints
from typing import Union, Tuple


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


def task_2a():
    """
    A method that satisfies task 2a

    Parameters
    ----------
    Does not accept any parameters

    Returns
    -------
    Does not return any value.
    """

    print(" I: Executing task 2a...")

    # Set up a range based on the smallest length
    least = min([LEN_H1, LEN_H2, LEN_H3])
    n = np.arange(0, least)

    # Plot the impulse responses
    plt.plot(n, h1[n], "-o", label="$h_1$")
    plt.plot(n, h2[n], "-o", label="$h_2$")
    plt.plot(n, h3[n], "-o", label="$h_3$")

    # - set up grid and legend
    plt.grid()
    plt.legend()

    # - use proper title and axis labels
    plt.title("Impulse response of FIR filters")
    plt.xlabel("Input $n$ [1]")
    plt.ylabel("Impulse response $h[n]$ [1]")

    # - show figure
    plt.show()


def task_2b():
    """
    A method that satisfies task 2b

    Parameters
    ----------
    Does not accept any parameters

    Returns
    -------
    Does not return any value
    """

    print(" I: Executing task 2b...")

    # Set up toy signal and impulse response
    x = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0])      # - rectangular pulse
    h = np.array([3, 2, 1, 0])                           # - triangular filter

    # Calculate same and full convolutions
    c_same = convolve(x, h, 0)
    c_full = convolve(x, h, 1)
    ref_same = np.convolve(x, h, "same")
    ref_full = np.convolve(x, h, "full")

    # Set up index arrays
    n_same = np.arange(len(x))
    n_full = np.arange(len(x) + len(h) - 1)

    # Plot calculated and reference values
    plt.plot(n_same, c_same, '-o', label="same")
    plt.plot(n_full, c_full, '-o', label="full")

    plt.plot(n_same, ref_same, '--o', label="same, reference")
    plt.plot(n_full, ref_full, '--o', label="full, reference")

    # Set up grid and legend
    plt.grid()
    plt.legend()

    # Show figure
    plt.show()


def task_2c():
    pass


def task_2d():
    pass


def task_2e():
    pass


def task_2f():
    pass


def task_2g():
    pass


def main():
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
    task_2a()
    task_2b()
    task_2c()
    task_2d()
    task_2e()
    task_2f()

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
