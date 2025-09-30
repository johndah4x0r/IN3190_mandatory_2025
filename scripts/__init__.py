#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A helper module containing scripts that
double as importable modules.
"""

# Copyright (c) 2025 John Isaac Calderon

# OS-related hooks
import os


# Checks whether the LHS file is older than the RHS file
# - returns `True` if `lhs` is older than `rhs`, or if
# `lhs` doesn't exist and `rhs` does
# - if neither files exist, then the result is always `False`
def is_older(lhs: str, rhs: str) -> bool:
    """
    Checks whether one file is older than the other

    Parameters
    ----------
    lhs : str
        Path to left-hand-side file ('older' candidate)
    rhs : str
        Path to right-hand-side file ('younger' candidate)

    Returns
    -------
    older : bool
        Status that indicates relative age

        Returns `True` if `lhs` is older than `rhs`, or if
        `lhs` doesn't and `rhs` does. Returns `False` if
        `rhs` is older than `lhs`, or if `rhs` doesn't and
        `lhs` does.

        If neither file exists, then `older = False`, as
        there's nothing to compare.
    """

    # Attempt to stat LHS and RHS
    try:
        st1 = os.stat(lhs)
    except FileNotFoundError:
        st1 = None
    try:
        st2 = os.stat(rhs)
    except FileNotFoundError:
        st2 = None

    if st1 is None and st2 is None:
        return False
    if st1 is None and st2 is not None:
        return True
    if st1 is not None and st2 is None:
        return False

    # Dynamically determine age, and return status
    older = st1.st_mtime < st2.st_mtime
    return older
