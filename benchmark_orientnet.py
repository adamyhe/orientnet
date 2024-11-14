import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from custom_loss import rescale_sigmoid

# Load experimental data

expt_fname = "/home2/ayh8/data/lcl/merged_procap_0.csv.gz"
expt = pd.read_csv(expt_fname, index_col=0, header=None)
expt_pl = expt.iloc[:, 250:750].sum(axis=1)
expt_mn = expt.iloc[:, 1250:1750].sum(axis=1)
expt_orientation = np.max([expt_pl, expt_mn], axis=0) / (expt_pl + expt_mn)

# Load predictions

pred_fname = "/home2/ayh8/data/lcl/merged_orientation_logits_index_0.npz"
pred_orientation = np.load(pred_fname)["arr_0"]

# CLIPNET
# PearsonRResult(statistic=0.6006778171560757, pvalue=0.0)

# ORIENTNET
# pearsonr(expt_orientation, pred_orientation.flatten())
# PearsonRResult(statistic=0.613683144358866, pvalue=0.0)
pearsonr(expt_orientation, np.array(rescale_sigmoid(pred_orientation)).flatten())
