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

from .auxiliary import *
from .variables import *

__all__ = ["save_vcf", "rezip_vcf", "save_lst"]


def save_vcf(df,
             metadata,
             rezip = True,
             prefix='chr',
             outdir=None,
             save_name=None
             ):  # Only use this if no cols are removed from the original vcf
    # df is the vcf_df to be saved
    # metadata is a list generated from read_metadata
    df = df.copy()

    if type(df.iloc[0, 0] == int):
        df[df.columns[0]] = prefix + df[df.columns[0]].astype(str)
    random_str = secrets.token_hex(8) + '_'

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    file_path = outdir + random_str + 'test.vcf'
    metadata_path = outdir + random_str + 'metadata.txt'

    df.to_csv(file_path, index=False, sep='\t', header=False)

    with open(metadata_path, 'w') as metadata_file:
        metadata_file.write(''.join(metadata))

    gzipped_file_path = outdir + save_name

    with open(metadata_path, 'rb') as metadata_file:
        metadata_content = metadata_file.read()

    with open(file_path, 'rb') as data_file, gzip.open(gzipped_file_path,
                                                       'wb') as gzipped_file:
        gzipped_file.write(metadata_content)
        gzipped_file.writelines(data_file)

    os.remove(file_path)
    os.remove(metadata_path)

    if rezip:
        rezip_vcf(gzipped_file_path)


def rezip_vcf(f, index=True):
    vcf = f.replace('.gz', '')

    clean = 'rm -f ' + vcf
    subprocess.run(clean, shell=True)

    decompress = 'gunzip ' + f
    subprocess.run(decompress, shell=True)

    recompress = 'bgzip ' + vcf
    subprocess.run(recompress, shell=True)

    if index:
        index = 'tabix -f -p vcf ' + f
        subprocess.run(index, shell=True)

    return None

def save_lst(file, lst):
    with open(file, 'w') as file:
        for item in lst:
            file.write("%s\n" % item)
    return None
