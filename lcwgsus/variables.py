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

COMMON_COLS = ['chr', 'pos', 'ref', 'alt']
VCF_COLS = [
    'chr', 'pos', 'ID', 'ref', 'alt', 'QUAL', 'FILTER', 'INFO', 'FORMAT'
]

LC_SAMPLE_PREFIX = 'GM'
CHIP_SAMPLE_PREFIX = 'GAM'
SEQ_SAMPLE_PREFIX = 'IDT'

SAMPLE_LINKER_FILE = 'data/metadata/sample_linker.csv'

MAF_ARY = np.array([
            0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
            0.1, 0.2, 0.5, 0.95, 1
        ])
