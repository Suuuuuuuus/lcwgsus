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

FV_IDT_NAMES_FILE = '/well/band/users/rbx225/GAMCC/data/sample_tsvs/fv_idt_names.tsv'
FV_GM_NAMES_FILE = '/well/band/users/rbx225/GAMCC/data/sample_tsvs/fv_gm_names.tsv'
FV_GAM_NAMES_FILE = '/well/band/users/rbx225/GAMCC/data/sample_tsvs/fv_gam_names.tsv'
MINI_IDT_NAMES_FILE = '/well/band/users/rbx225/GAMCC/data/sample_tsvs/mini_idt_names.tsv'
MINI_GM_NAMES_FILE = '/well/band/users/rbx225/GAMCC/data/sample_tsvs/mini_gm_names.tsv'
MINI_GAM_NAMES_FILE = '/well/band/users/rbx225/GAMCC/data/sample_tsvs/mini_gam_names.tsv'

SAMPLE_LINKER_FILE = '/well/band/users/rbx225/GAMCC/data/metadata/sample_linker.csv'
G_CODE_FILE = '/well/band/users/rbx225/GAMCC/data/hla_direct_sequencing/hla_nom_g.txt'
AMBIGUOUS_G_CODE_FILE = '/well/band/users/rbx225/GAMCC/data/hla_direct_sequencing/ambiguous_G_codes.tsv'

MAF_ARY = np.array([
            0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
            0.1, 0.2, 0.5, 0.95, 1
        ])

CASE_CONTROLS = ['non-malaria_control', 'mild_malaria', 'severe_malaria']
ETHNICITIES = ['fula', 'jola', 'mandinka', 'wollof']

HLA_GENES = ['A', 'B', 'C', 'DQB1', 'DRB1']
HLA_GENES_ALL = ['A', 'B', 'C', 'DMA', 'DMB', 'DOA', 'DOB', 'DPA1', 'DPB1', 'DPB2', 'DQA1', 'DQA2', 'DQB1', 'DRA', 'DRB1', 'DRB5', 'E', 'F', 'G', 'HFE', 'H', 'J', 'L', 'MICA', 'MICB', 'TAP1', 'TAP2', 'V']
HLA_GENES_ALL_EXPANDED = ['A', 'B', 'C', 'DMA', 'DMB', 'DOA', 'DOB', 'DPA1', 'DPA2', 'DPB1', 'DPB2', 'DQA1', 'DQA2', 'DQB1', 'DQB2', 'DRA', 'DRB1', 'DRB3', 'DRB4', 'DRB5', 'E', 'F', 'G', 'H', 'HFE', 'J', 'K', 'L', 'MICA', 'MICB', 'N', 'P', 'R', 'S', 'T', 'TAP1', 'TAP2', 'U', 'V', 'W', 'Y']
HLA_LOCI = [i+'1' for i in HLA_GENES] + [i+'2' for i in HLA_GENES]

HLA_DIRECT_SEQUENCING_FILE = '/well/band/users/rbx225/GAMCC/data/hla_direct_sequencing/HLA_direct_sequencing_all.csv'

COLORBAR_CMAP_STR = 'GnBu'
COLORBAR_CMAP = plt.get_cmap(COLORBAR_CMAP_STR)
COLORBAR_CMAP_HEX = [mcolors.rgb2hex(COLORBAR_CMAP(i)) for i in range(COLORBAR_CMAP.N)]

CATEGORY_CMAP_STR = 'tab20'
CATEGORY_CMAP = plt.get_cmap(CATEGORY_CMAP_STR)
CATEGORY_CMAP_HEX = [mcolors.rgb2hex(CATEGORY_CMAP(i)) for i in range(CATEGORY_CMAP.N)]