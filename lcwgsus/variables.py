import io
import os
import sys
import csv
import gzip
import time
import random
import secrets
import subprocess
import resource
import itertools
import multiprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import statsmodels.api as sm
import scipy
from typing import Union, Tuple, List
from scipy.stats import poisson
from scipy.stats import chi2
from scipy.stats import friedmanchisquare
from scipy.stats import studentized_range

pd.options.mode.chained_assignment = None

CHROMOSOMES_ALL = [str(i) for i in range(1,23)]

COMMON_COLS = ['chr', 'pos', 'ref', 'alt']
VCF_COLS = [
    'chr', 'pos', 'ID', 'ref', 'alt', 'QUAL', 'FILTER', 'INFO', 'FORMAT'
]
IMPACC_METRICS = ['NRC', 'r2', 'ccd_homref', 'ccd_het', 'ccd_homalt']
IMPACC_COLS = [
    'NRC', 'NRC_BC', 'r2', 'r2_BC', 'ccd_homref', 'ccd_homref_BC', 'ccd_het',
    'ccd_het_BC', 'ccd_homalt', 'ccd_homalt_BC'
]
IMPACC_H_COLS = [
    'n_variants', 'NRC', 'NRC_BC', 'NRC_AC', 'r2', 'r2_BC', 'r2_AC',
    'ccd_homref', 'ccd_homref_BC', 'ccd_homref_AC', 'ccd_het', 'ccd_het_BC',
    'ccd_het_AC', 'ccd_homalt', 'ccd_homalt_BC', 'ccd_homalt_AC'
]
IMPACC_V_COLS = [
    'sample', 'AF', 'n_variants', 'NRC', 'NRC_BC', 'NRC_AC', 'r2', 'r2_BC',
    'r2_AC', 'ccd_homref', 'ccd_homref_BC', 'ccd_homref_AC', 'ccd_het',
    'ccd_het_BC', 'ccd_het_AC', 'ccd_homalt', 'ccd_homalt_BC', 'ccd_homalt_AC'
]


LC_SAMPLE_PREFIX = 'GM'
CHIP_SAMPLE_PREFIX = 'GAM'
SEQ_SAMPLE_PREFIX = 'IDT'

SAMPLE_LINKER_FILE = 'data/metadata/sample_linker.csv'

MAF_ARY = np.array([
            0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
            0.1, 0.2, 0.5, 0.95, 1
        ])

CASE_CONTROLS = ['non-malaria_control', 'mild_malaria', 'severe_malaria']
ETHNICITIES = ['fula', 'jola', 'mandinka', 'wollof']


def generate_af_axis(x=MAF_ARY):
    x = [
        str(int(i)) if i == int(i) else str(float(i)).rstrip('0').rstrip('.')
        for i in x * 100
    ]
    y = x[:-1]
    res_ary = list(x[0])
    shift = x[1:]
    combine = [i + '-' + j for i, j in zip(y, shift)]
    return combine

AF_AXIS = generate_af_axis(MAF_ARY)