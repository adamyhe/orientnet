import h5py
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Load experimental data

expt_fname = "/home2/ayh8/data/lcl/merged_procap_0.csv.gz"
expt = pd.read_csv(expt_fname, index_col=0, header=None)
expt_pl = expt.iloc[:, 250:750].sum(axis=1)
expt_mn = expt.iloc[:, 1250:1750].sum(axis=1)
expt_orientation = np.max([expt_pl, expt_mn], axis=0) / (expt_pl + expt_mn)
expt_direction = (expt_pl + 1e-6) / (expt_mn + 1e-6)

# Load predictions

pred_fname = "/home2/ayh8/data/lcl/merged_reference_prediction.h5"
with h5py.File(pred_fname, "r") as f:
    pred_pl = f["track"][:, :500].sum(axis=1)
    pred_mn = f["track"][:, 500:].sum(axis=1)
    pred_orientation = np.max([pred_pl, pred_mn], axis=0) / (pred_pl + pred_mn)
    pred_direction = (pred_pl + 1e-6) / (pred_mn + 1e-6)

pearsonr(expt_orientation, pred_orientation)
# PearsonRResult(statistic=0.6006778171560757, pvalue=0.0)

spearmanr(expt_direction, pred_direction)
# SignificanceResult(statistic=0.7838727838669889, pvalue=0.0)

pearsonr(np.log(expt_direction), np.log(pred_direction))
# PearsonRResult(statistic=0.7415347551091522, pvalue=0.0)
