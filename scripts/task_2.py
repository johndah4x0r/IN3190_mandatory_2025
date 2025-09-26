#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A script that satisfies task 2 (Data filtering)
"""

# Import FIR filters
from .fir import h1, h2, h3, LEN_H1, LEN_H2, LEN_H3

# Import modules for general analysis
import numpy as np
import matplotlib.pyplot as plt

# Copyright (c) 2025 John Isaac Calderon


def task_2a(strict: bool = False):
    """
    A method that satisfies task 2a

    Accepts:
        strict: bool
            Determines whether assertions are performed

    Does not return any value.
    """

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


def convolve(x: np.ndarray, h: np.ndarray, ylen_choice: int) -> np.ndarray:
    """
    Performs a convolution between `x` and `h`

    Accepts:
        x: np.ndarray
            An (M,)-shaped array
        h: np.ndarray
            An (N,)-shaped array
        ylen_choice: int
            Determines convolution mode

            If `ylen_choice == 0`, then the output will have
            the shape (M,), whereas if `ylen_choice == 1`,
            then the output will have the shape (M + N - 1).

    Returns:
        np.ndarray - the convolution of `x` and `h`
    """

    # Even if the task doesn't demand it, we'll try
    # to speed up the convolution process by using
    # whatever trick we can come up with

    # - alias `x` and `h` to vectorized functions
    _x = np.vectorize(lambda n: x[n] if 0 <= n <= len(x) else 0)
    _h = np.vectorize(lambda n: h[n] if 0 <= n <= len(h) else 0)

    # - determine length, and obtain input array
    l = len(x) + len(h) - 1 if ylen_choice else len(x)
    n = np.arange(0, l)

    # - calculate products, then return sum of products
    prods 
def task_2b():