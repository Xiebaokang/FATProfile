#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch # Import the Patch class
from matplotlib.lines import Line2D
from enum import Enum
from typing import Tuple, Dict, List, Callable
from dataclasses import dataclass
import matplotlib.ticker as mticker
import copy

import json
import csv

from runner_json import SDPAVariant, SDPACompareResultEntry, PrecisionTestResult, LoopPrecisionTestResult


_THIS_SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

#################### General Utilities BEGIN #####################

#################### General Utilities END #####################


#################### Datafile Defs and Utilities BEGIN #####################

DATAFILE = _THIS_SCRIPT_DIR / "loop_sdpa_precision_results.json"

def load_loop_precision_test_result(file_path: pathlib.Path) -> LoopPrecisionTestResult:
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    return LoopPrecisionTestResult.from_json(json_data)

def func_extract_mse(result: SDPACompareResultEntry) -> float:
    return result.mse

def func_extract_rmse(result: SDPACompareResultEntry) -> float:
    return result.rmse

def gen_error_metric_series(
    ref_method: SDPAVariant,
    target_method: SDPAVariant,
    f_extract_metric: Callable[[SDPACompareResultEntry], float],
    min_outlier_variance_inclusive: float,
    max_outlier_variance_inclusive: float,
    loop_precision_test_result: LoopPrecisionTestResult
) -> Tuple[List[float], List[float]]: # return (x: outlier_variance_list, y: error_metric_list)
    outlier_vriance_list = []
    error_metric_list = []

    for result in loop_precision_test_result.results:
        result_outlier_variance = result.outlier_variance
        # skip unwanted outlier variances
        if (result_outlier_variance < min_outlier_variance_inclusive or
            result_outlier_variance > max_outlier_variance_inclusive):
            continue
        for compare_result in result.compare_results:
            if (compare_result.ref_method == ref_method and
                compare_result.target_method == target_method):
                outlier_vriance_list.append(result_outlier_variance)
                error_metric_list.append(f_extract_metric(compare_result))
    
    # sort the lists by outlier variance in ascending order
    sorted_indices = np.argsort(outlier_vriance_list)
    outlier_vriance_list = [outlier_vriance_list[i] for i in sorted_indices]
    error_metric_list = [error_metric_list[i] for i in sorted_indices]

    return outlier_vriance_list, error_metric_list


#################### Datafile Defs and Utilities END #####################

# plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 8 # Global default for most text elements

plt.rcParams['axes.labelsize'] = 12       # Size of x and y axis labels of the whole figure
plt.rcParams['xtick.labelsize'] = 10     # Size of x-axis tick labels
plt.rcParams['ytick.labelsize'] = 10      # Size of y-axis tick labels
plt.rcParams['legend.fontsize'] = 12      # Size of legend entries
plt.rcParams['axes.titlesize'] = 12       # Size of subplot titles


SINGLE_FIG_WIDTH_INCHES = 7
SINGLE_FIG_HEIGHT_INCHES = 2.4
USE_TITLE = False

class SDPAErrorSeriesType(Enum):
    FP64_MATH_VS_FP16_MATH = "FP16 Math"
    FP64_MATH_VS_FP16_FA = "FP16 FA2"
    FP64_MATH_VS_FP16_FULLSIMT = "FP16 FA2+Max16"
    FP64_MATH_VS_FP16_FULLMMA = "FP16 AllTensor"
    FP64_MATH_VS_FP16_ILPH = "FP16 FA-T"
    FP64_MATH_VS_FP16_ILPV = "FP16 Vertical Split ILP"
    FP16_FA_VS_FP16_FULLSIMT = "FP16 Max Surrogate Only"
    FP16_FA_VS_FP16_FULLMMA = "FP16 All Tensorized"
    FP16_FA_VS_FP16_ILPH = "FP16 Horizontal Split ILP"
    FP16_FA_VS_FP16_ILPV = "FP16 Vertical Split ILP"

# (ref, target, metric_func, min_outlier_variance_inclusive, max_outlier_variance_exclusive)
SDPAErrorSeriesDict_SetStandard = {
    # SDPAErrorSeriesType.FP64_MATH_VS_FP16_MATH: (SDPAVariant.FP64_MATH, SDPAVariant.FP16_MATH, func_extract_rmse, 0, 100),
    SDPAErrorSeriesType.FP64_MATH_VS_FP16_FA: (SDPAVariant.FP64_MATH, SDPAVariant.FP16_FA, func_extract_rmse, 0, 100),
    SDPAErrorSeriesType.FP64_MATH_VS_FP16_FULLSIMT: (SDPAVariant.FP64_MATH, SDPAVariant.FP16_FULLSIMT, func_extract_rmse, 0, 100),
    # SDPAErrorSeriesType.FP64_MATH_VS_FP16_FULLMMA: (SDPAVariant.FP64_MATH, SDPAVariant.FP16_FULLMMA, func_extract_rmse, 0, 100),
    SDPAErrorSeriesType.FP64_MATH_VS_FP16_ILPH: (SDPAVariant.FP64_MATH, SDPAVariant.FP16_ILPH, func_extract_rmse, 0, 100),
    # SDPAErrorSeriesType.FP64_MATH_VS_FP16_ILPV: (SDPAVariant.FP64_MATH, SDPAVariant.FP16_ILPV, func_extract_rmse, 0, 100)
}

SDPAErrorSeriesColorMap = {
    SDPAErrorSeriesType.FP64_MATH_VS_FP16_MATH: "green",
    SDPAErrorSeriesType.FP64_MATH_VS_FP16_FA: "blue",
    SDPAErrorSeriesType.FP64_MATH_VS_FP16_FULLSIMT: "olive",
    SDPAErrorSeriesType.FP64_MATH_VS_FP16_FULLMMA: "darkorange",
    SDPAErrorSeriesType.FP64_MATH_VS_FP16_ILPH: "red",
    SDPAErrorSeriesType.FP64_MATH_VS_FP16_ILPV: "purple",
    SDPAErrorSeriesType.FP16_FA_VS_FP16_FULLSIMT: "olive",
    SDPAErrorSeriesType.FP16_FA_VS_FP16_FULLMMA: "darkorange",
    SDPAErrorSeriesType.FP16_FA_VS_FP16_ILPH: "red",
    SDPAErrorSeriesType.FP16_FA_VS_FP16_ILPV: "purple"
}

SDPAErrorSeriesAlpha = 0.9

def plot_sdpa_error_fig(sdpa_error_series_dict: Dict, error_series_dict_filetag: str):
    
    fig, ax = plt.subplots(figsize=(SINGLE_FIG_WIDTH_INCHES, SINGLE_FIG_HEIGHT_INCHES))

    for error_series_type, (ref_method, target_method, f_extract_metric, min_outlier_variance, max_outlier_variance) in sdpa_error_series_dict.items():
        # Load the loop precision test result
        loop_precision_test_result = load_loop_precision_test_result(DATAFILE)

        # Generate the error metric series
        variance_list, error_metric_list = gen_error_metric_series(
            ref_method,
            target_method,
            f_extract_metric,
            min_outlier_variance,
            max_outlier_variance,
            loop_precision_test_result
        )

        seires_color = SDPAErrorSeriesColorMap[error_series_type]

        if error_series_type is SDPAErrorSeriesType.FP64_MATH_VS_FP16_FA:
            ax.plot(
                variance_list,
                error_metric_list,
                label=error_series_type.value,
                color=seires_color,
                # linestyle='--',
                # linewidth=3,
                alpha=SDPAErrorSeriesAlpha
            )
        else:
            ax.plot(
                variance_list,
                error_metric_list,
                label=error_series_type.value,
                color=seires_color,
                alpha=SDPAErrorSeriesAlpha
            )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(
        'Outlier Variance',
        # fontsize=10
    )
    ax.set_ylabel(
        'RMSE (lower is better)',
        # fontsize=10
    )

    AXVLINE_X = 52.0
    ax.axvline(
        x = AXVLINE_X,
        color = 'black',
        linestyle = '--',
        linewidth = 1.0,
    )
    
    # set desired txicks
    desired_xticks = [1.0, 10.0, AXVLINE_X, 100.0]
    ax.set_xticks(desired_xticks)
    # set desired xtick labels
    ax.set_xticklabels([f"{tick:.1f}" for tick in desired_xticks])

     # xlim
    ax.set_xlim(left=1.0, right=100.0)

    # legend
    ax.legend(
        # loc='upper center',
        # bbox_to_anchor=(0.5, 1.05),
        ncol=1,
        # fontsize=10,
        # frameon=False,
        # handlelength=1.5,
        # handletextpad=0.5,
        # columnspacing=1.0
    )

    # grid
    ax.grid(which='major', axis='both', linestyle=':', linewidth=1.2, alpha=0.6)
    ax.grid(which='minor', axis='both', linestyle=':', linewidth=0.8, alpha=0.3)


    

    if USE_TITLE:
        ax.set_title("A100 FP16-FP32, hdim 128, seqlen 4096")
        # FOR PLOT WITH TITLE
        plt.subplots_adjust(
            top = 0.850,
            bottom = 0.210,
            left = 0.103,
            right = 0.966,
            hspace = 0.316,
            wspace = 0.140
        )
    else:
        ax.set_title("")
        # Fine-tune this with GUI
        # FOR PLOT WITHOUT TITLE
        plt.subplots_adjust(
            top = 0.900,
            bottom = 0.190,
            left = 0.103,
            right = 0.966,
            hspace = 0.431,
            wspace = 0.140
        )
        plt.subplots_adjust(
            top = 0.965,
            bottom = 0.210,
            left = 0.103,
            right = 0.966,
            hspace = 0.316,
            wspace = 0.140
        )

    fig.savefig(
        _THIS_SCRIPT_DIR / f'fig-standalone-fa2-rmse-{error_series_dict_filetag}.pdf',
        bbox_inches='tight',
        pad_inches=0.01
    )
    # plt.show()



plot_sdpa_error_fig(
    SDPAErrorSeriesDict_SetStandard,
    "standard"
)