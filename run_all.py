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

# Import scripts
import scripts
