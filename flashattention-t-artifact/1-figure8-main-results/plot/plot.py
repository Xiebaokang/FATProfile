#! /usr/bin/env python3

import os
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch # Import the Patch class
from enum import Enum
from typing import Tuple, Dict, List
from dataclasses import dataclass
import json
import csv


_THIS_SCRIPT_DIR = pathlib.Path(__file__).parent.resolve().absolute()



#################### General Utilities BEGIN #####################

class Machine(Enum):
    H100 = "H100"
    A100 = "A100"

class Method(Enum):
    OURS = "Ours"
    FA2 = "FlashAttention-2"
    FA3 = "FlashAttention-3"
    TRITON = "Triton"
    XFORMERS = "xFormers"
    FLASHINFER = "FlashInfer"

def get_hdim_causal_tuple(headdim: int, causal: bool) -> Tuple[int, bool]:
    return (headdim, causal)

# calculate the FLOP number for attention operations, ONLY considering the GEMM FLOPs.
def calc_attention_flops_gemm_only(batchsize:int, nheads:int, seqlen:int, headdim:int, is_causal:bool) -> int:
    # f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return 4 * batchsize * seqlen**2 * nheads * headdim // (2 if is_causal else 1)

# calculate the FLOP number for attention operations, WITH all the vector operations included.
def calc_attention_flops_full(batchsize:int, nheads:int, seqlen:int, headdim:int, is_causal:bool) -> int:
    gemm_flops = 4 * batchsize * seqlen**2 * nheads * headdim
    ffma_flops = 2 * seqlen**2 * nheads * batchsize
    fadd_flops = seqlen**2 * nheads * batchsize
    fmul_flops = seqlen * headdim * nheads * batchsize
    total_flops = (gemm_flops + ffma_flops + fadd_flops) // (2 if is_causal else 1) + fmul_flops
    return total_flops

# calculate the TFLOPS, given number of FLOPs and time in milliseconds.
def calc_tflops(flops: int, time_ms: float) -> float:
    if time_ms <= 0:
        raise ValueError("Time in milliseconds must be greater than zero.")
    return flops / (time_ms * 1e9)  # Convert ms to seconds and calculate TFLOPS

#################### General Utilities END #####################


#################### Datafile Defs and Utilities BEGIN #####################


H100_OURS_DATA_FILEPATH = _THIS_SCRIPT_DIR / "data_ours_h100.csv"
H100_FA3_DATA_FILEPATH = _THIS_SCRIPT_DIR / "data_fa3_h100.csv"
H100_TRITON_DATA_FILEPATH = _THIS_SCRIPT_DIR / "data_triton_h100.csv"
H100_FLASHINFER_DATA_FILEPATH = _THIS_SCRIPT_DIR / "data_flashinfer_h100.csv"

A100_OURS_DATA_FILEPATH = _THIS_SCRIPT_DIR / "data_ours_a100.csv"
A100_FA2_DATA_FILEPATH = _THIS_SCRIPT_DIR / "data_fa2_a100.csv"
A100_TRITON_DATA_FILEPATH = _THIS_SCRIPT_DIR / "data_triton_a100.csv"
A100_FLASHINFER_DATA_FILEPATH = _THIS_SCRIPT_DIR / "data_flashinfer_a100.csv"
# dual-level dictionary for querying datafile path using [machine][method]
# level1: machine
# level2: method
DatafileMachineMethodDict = {
    Machine.A100: {
        Method.OURS: A100_OURS_DATA_FILEPATH,
        Method.FA2: A100_FA2_DATA_FILEPATH,
        Method.TRITON: A100_TRITON_DATA_FILEPATH,
        Method.FLASHINFER: A100_FLASHINFER_DATA_FILEPATH
    },
    Machine.H100: {
        Method.OURS: H100_OURS_DATA_FILEPATH,
        Method.FA3: H100_FA3_DATA_FILEPATH,
        Method.TRITON: H100_TRITON_DATA_FILEPATH,
        Method.FLASHINFER: H100_FLASHINFER_DATA_FILEPATH
    }
}

@dataclass
class DataEntry:
    dataType: str
    comment: str
    batchsize: int
    nheads: int
    seqlen: int
    headdim: int
    is_causal: bool
    time_ms: float


# load the csv datafile into a list of DataEntries
# NOTE: the list is sorted into an ascending order w.r.t. seqlen
def load_csv_datafile(machine: Machine, method: Method) -> List[DataEntry]:
    datafile_path = DatafileMachineMethodDict[machine][method]
    if not datafile_path.exists():
        raise FileNotFoundError(f"Data file not found: {datafile_path}")
    
    def parse_bool(x: str) -> bool:
        int_x = int(x)
        if int_x == 0 or int_x == 1:
            return bool(int_x)
        elif x.lower() in ['true', 'false']:
            return x.lower() == 'true'
        else:
            raise ValueError(f"Invalid boolean value: {x}")
    
    data_entries = []
    with open(datafile_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert the row to a DataEntry object
            entry = DataEntry(
                dataType=row['DataType'],
                comment=row['Comment'],
                batchsize=int(row['batchsize']),
                nheads=int(row['nheads']),
                seqlen=int(row['seqlen']),
                headdim=int(row['headdim']),
                is_causal=parse_bool(row['is_causal']),
                time_ms=float(row['time_ms'])
            )
            data_entries.append(entry)

    # sort to ascending order respect to seqlen
    data_entries = sorted(data_entries, key=lambda x: x.seqlen)

    return data_entries



#################### Datafile Defs and Utilities END #####################

# plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 8 # Global default for most text elements

plt.rcParams['axes.labelsize'] = 13       # Size of x and y axis labels of the whole figure
# plt.rcParams['xtick.labelsize'] = 12     # Size of x-axis tick labels
# plt.rcParams['ytick.labelsize'] = 11      # Size of y-axis tick labels
xtick_labelsize = 10 # Size of x-axis tick labels
ytick_labelsize = 9 # Size of y-axis tick labels
plt.rcParams['legend.fontsize'] = 12      # Size of legend entries
plt.rcParams['axes.titlesize'] = 12       # Size of subplot titles
plt.rcParams['figure.titlesize'] = 10    # Size of suptitle (overall figure title)
annotation_fontsize = 7 # Size of annotations on the plot

SINGLE_FIG_WIDTH_INCHES = 15   # 7 inches for acmart full-width (2-col) figure
SINGLE_FIG_HEIGHT_INCHES = 7.5 # change this if necessary

SUBPLOTS_NUM_ROWS = 2
SUBPLOTS_NUM_COLS = 4

MachinePrecisionDict = {
    Machine.A100: "FP16-FP32",
    Machine.H100: "FP8-FP32"
}

# defines the row index for each machine in the subplot grid
MachineRowIndexDict = {
    Machine.A100: 0,
    Machine.H100: 1
}

# defines the column index for each (headdim, causal) tuple in the subplot grid
HdimCasualColIndexDict = {
    get_hdim_causal_tuple(64, False): 0,
    get_hdim_causal_tuple(64, True): 1,
    get_hdim_causal_tuple(128, False): 2,
    get_hdim_causal_tuple(128, True): 3
}

# defines the MAJOR ytick base (interval) for each (machine, (headdim, causal)) tuple in the subplot grid
SubplotYtickMajorBaseDict = {
    (Machine.A100, get_hdim_causal_tuple(64, False)):   50,
    (Machine.A100, get_hdim_causal_tuple(64, True)):    50,
    (Machine.A100, get_hdim_causal_tuple(128, False)):  50,
    (Machine.A100, get_hdim_causal_tuple(128, True)):   50,
    (Machine.H100, get_hdim_causal_tuple(64, False)):   100,
    (Machine.H100, get_hdim_causal_tuple(64, True)):    100,
    (Machine.H100, get_hdim_causal_tuple(128, False)):  100,
    (Machine.H100, get_hdim_causal_tuple(128, True)):   100
}

# defines the MINOR ytick base (interval) for each (machine, (headdim, causal)) tuple in the subplot grid
SubplotYtickMinorBaseDict = {
    (Machine.A100, get_hdim_causal_tuple(64, False)):   25,
    (Machine.A100, get_hdim_causal_tuple(64, True)):    25,
    (Machine.A100, get_hdim_causal_tuple(128, False)):  25,
    (Machine.A100, get_hdim_causal_tuple(128, True)):   25,
    (Machine.H100, get_hdim_causal_tuple(64, False)):   50,
    (Machine.H100, get_hdim_causal_tuple(64, True)):    50,
    (Machine.H100, get_hdim_causal_tuple(128, False)):  50,
    (Machine.H100, get_hdim_causal_tuple(128, True)):   50
}



# The central list for method ordering
# for BOTH the the bar plot and the legend
MethodBarOrderList = [
    Method.TRITON,
    Method.FLASHINFER,
    Method.FA2,  # FA2 is used for A100
    Method.FA3,  # FA3 is used for H100
    Method.OURS
]

def get_actual_method_orders(methods:List[Method]) -> Dict[Method, int]:
    # sort the method based on the order indices
    sorted_methods = sorted(methods, key=lambda x: MethodBarOrderList.index(x))
    # create a mapping from method to its order index
    method_order_dict = {method: i for i, method in enumerate(sorted_methods)}
    return method_order_dict

MethodBarWidth = 0.2
MethodBarAlpha = 0.75

# This map also defines what methods are actually plotted
MethodColorMap = {
    Method.FLASHINFER: "orange",
    Method.TRITON: "purple",
    Method.FA2: "blue",
    Method.FA3: "green",
    Method.OURS: "red"
}

def inverse_query_row_index_machine(row_index: int) -> Machine:
    for machine, index in MachineRowIndexDict.items():
        if index == row_index:
            return machine
    raise ValueError(f"Invalid row index: {row_index}")

def inverse_query_col_index_hdim_causal(col_index: int) -> Tuple[int, bool]:
    for hdim_causal, index in HdimCasualColIndexDict.items():
        if index == col_index:
            return hdim_causal
    raise ValueError(f"Invalid column index: {col_index}")


def func_seqlen_filter(data_entry: DataEntry) -> bool:
    used_seqlens = [512, 1024, 2048, 4096, 8192]
    threshold_ratio = 0.3
    
    used_seqlens_np = np.array(used_seqlens)
    cond = np.any(np.abs(used_seqlens_np - data_entry.seqlen) <= threshold_ratio * used_seqlens_np)
    return cond

def get_seqlen_base_xvalue_and_xticklabel_list(data_entry_list: List[DataEntry]) -> Tuple[List[int], List[str]]:
    xvalues = [i for i in range(len(data_entry_list))]
    label_dict = {
        128: "128",
        256: "256",
        512: "512",
        1024: "1k",
        2048: "2k",
        4096: "4k",
        8192: "8k",
        16384: "16k",
        32768: "32k",
        65536: "64k"
    }

    label_tick_values_np = np.array([label_value for label_value, label_str in label_dict.items()])


    threshold_ratio = 0.3
    seqlen_list = [data_entry.seqlen for data_entry in data_entry_list]
    seqlen_label_list = []
    for seqlen in seqlen_list:
        thresholds = label_tick_values_np * threshold_ratio
        abs_diffs = np.abs(label_tick_values_np - seqlen)
        conds = abs_diffs <= thresholds
        # assert that only one condition is True
        if np.sum(conds) != 1:
            raise ValueError(f"Multiple conditions matched for seqlen {seqlen}. Conditions: {conds}")
        matched_index = np.where(conds)[0][0]
        seqlen_label_list.append(label_dict[label_tick_values_np[matched_index]])
    return xvalues, seqlen_label_list
    
fig, axes = plt.subplots(SUBPLOTS_NUM_ROWS, SUBPLOTS_NUM_COLS, figsize=(SINGLE_FIG_WIDTH_INCHES, SINGLE_FIG_HEIGHT_INCHES))

for row_idx in range(SUBPLOTS_NUM_ROWS):
    for col_idx in range(SUBPLOTS_NUM_COLS):
        ax = axes[row_idx, col_idx]

        cur_subplot_machine = inverse_query_row_index_machine(row_idx)
        cur_subplot_hdim, cur_subplot_causal = inverse_query_col_index_hdim_causal(col_idx)
        is_ampere = cur_subplot_machine is Machine.A100
        is_hopper = cur_subplot_machine is Machine.H100


        _fa_method = Method.FA2 if is_ampere else Method.FA3
        data_entry_list_dict = {
            Method.FLASHINFER: load_csv_datafile(cur_subplot_machine, Method.FLASHINFER),
            Method.TRITON: load_csv_datafile(cur_subplot_machine, Method.TRITON),
            Method.OURS: load_csv_datafile(cur_subplot_machine, Method.OURS),
            _fa_method: load_csv_datafile(cur_subplot_machine, _fa_method)
        }
        actual_methods = list(data_entry_list_dict.keys())

        # filter out the data entry list based on hdim and causal
        func_hdim_causal_filter = lambda x: x.headdim == cur_subplot_hdim and x.is_causal == cur_subplot_causal
        for method, data_entry_list in data_entry_list_dict.items():
            data_entry_list_dict[method] = list(filter(func_hdim_causal_filter, data_entry_list))

        # apply custom filters
        # seqlen filter
        for method, data_entry_list in data_entry_list_dict.items():
            data_entry_list_dict[method] = list(filter(func_seqlen_filter, data_entry_list))

        # compute tflops
        def to_tflops_list(data_entry_list: List[DataEntry]) -> List[float]:
            tflops_list = []
            for entry in data_entry_list:
                flops = calc_attention_flops_full(
                    batchsize=entry.batchsize,
                    nheads=entry.nheads,
                    seqlen=entry.seqlen,
                    headdim=entry.headdim,
                    is_causal=entry.is_causal
                )
                tflops = calc_tflops(flops, entry.time_ms)
                tflops_list.append(tflops)
            return tflops_list
        
        tflops_list_dict = {}
        for method, data_entry_list in data_entry_list_dict.items():
            tflops_list_dict[method] = to_tflops_list(data_entry_list)

        # base xvalues and xtick labels
        base_xvalues_list_dict = {}
        xtick_labels_list_dict = {}
        for method, data_entry_list in data_entry_list_dict.items():
            base_xvalues, xtick_labels = get_seqlen_base_xvalue_and_xticklabel_list(data_entry_list)
            base_xvalues_list_dict[method] = base_xvalues
            xtick_labels_list_dict[method] = xtick_labels
        
        # apply offsets to the xvalues according to method order and barwidth
        order_dict = get_actual_method_orders(actual_methods)
        for method, base_xvalues in base_xvalues_list_dict.items():
            offset = order_dict[method] * MethodBarWidth
            base_xvalues_list_dict[method] = [x + offset for x in base_xvalues]
        

        # plot the bars
        for method, tflops_list in tflops_list_dict.items():
            method_color = MethodColorMap[method]
            base_xvalues = base_xvalues_list_dict[method]
            xtick_labels = xtick_labels_list_dict[method]
            ax.bar(
                base_xvalues,
                tflops_list,
                width=MethodBarWidth,
                label=method.value,
                alpha=MethodBarAlpha,  # Adjust the transparency of the bars
                color=method_color,
                # edgecolor='black'  # Add a border to the bars for better visibility
            )
            # annotate the bars with the tflops value directly on top of the bar
            for x, tflops in zip(base_xvalues, tflops_list):
                ax.text(
                    x, 
                    tflops + 0.01,  # Offset the text slightly above the bar
                    f"{tflops:.0f}",  # Format the tflops value to 0 decimal places
                    ha='center', 
                    va='bottom',
                    fontsize=annotation_fontsize,  # Adjust font size for readability
                    rotation=0  # Rotate the text for better visibility
                )

        # xticks and xtick labels
        # set the x-ticks to the middle of the bar clusters
        xticks = []
        num_plotted_methods = len(tflops_list_dict)
        method_order_0 = list(order_dict.keys())[0]  # the first method in the order
        method_0_xticks = base_xvalues_list_dict[method_order_0]
        xticks = [x + (num_plotted_methods - 1) * MethodBarWidth / 2 for x in method_0_xticks]
        ax.set_xticks(xticks)
        xtick_labels = xtick_labels_list_dict[method_order_0]
        ax.set_xticklabels(
            xtick_labels,
            fontsize = xtick_labelsize
            # fontsize=8
        )

        # set the title
        cur_subplot_machine_str = cur_subplot_machine.value
        cur_subplot_precision_str = MachinePrecisionDict[cur_subplot_machine]
        cur_subplot_hdim_str = f"hdim {cur_subplot_hdim}"
        # cur_subplot_causal = "causal" if cur_subplot_causal else "non-causal"
        cur_subplot_causal_str = "with causal mask" if cur_subplot_causal else "without causal mask"
        cur_subplot_causal_str = "causal" if cur_subplot_causal else "non-causal"
        cur_subplot_title_str = f"{cur_subplot_machine_str} {cur_subplot_precision_str}, {cur_subplot_hdim_str}, {cur_subplot_causal_str}"
        # valid fontweight options are: 'normal', 'bold', 'heavy', 'light', 'ultrabold', 'ultralight'
        ax.set_title(
            cur_subplot_title_str,
            # fontsize=10,
            fontweight='normal'
        )

        # ylim
        ymin_scale = 0.9
        ymax_scale = 1.05
        min_tflops = min([min(tflops_list) for tflops_list in tflops_list_dict.values()])
        max_tflops = max([max(tflops_list) for tflops_list in tflops_list_dict.values()])
        ax.set_ylim(
            ymin=min_tflops * ymin_scale,
            ymax=max_tflops * ymax_scale
        )

        if col_idx == 0:
            ax.set_ylabel(
                "TFLOPS",
                # fontsize=10
            )
        else:
            ax.set_ylabel("")

        # x label
        if row_idx == SUBPLOTS_NUM_ROWS - 1:
            ax.set_xlabel(
                "Sequence Length",
                # fontsize=10
            )
        else:
            ax.set_xlabel("")

        cur_subplot_y_major_base = SubplotYtickMajorBaseDict[(cur_subplot_machine, get_hdim_causal_tuple(cur_subplot_hdim, cur_subplot_causal))]
        cur_subplot_y_minor_base = SubplotYtickMinorBaseDict[(cur_subplot_machine, get_hdim_causal_tuple(cur_subplot_hdim, cur_subplot_causal))]
        ax.yaxis.set_major_locator(ticker.MultipleLocator(base=cur_subplot_y_major_base))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=cur_subplot_y_minor_base))
        ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())

        ax.tick_params(which='major', axis='y', length=4, labelsize=ytick_labelsize)   # Major tick labels
        ax.tick_params(which='minor', axis='y', length=4, labelsize=ytick_labelsize)   # Minor tick labels
  
        # grid
        ax.set_axisbelow(True)  # Ensure grid lines are below the data
        ax.grid(which='major', axis='y', linestyle=':', linewidth=1.2, alpha=0.5)  # Add light grid lines
        ax.grid(which='minor', axis='y', linestyle=':', linewidth=1.2, alpha=0.25)


# draw the legend on top of all subplots, using a horizontal layout
# NOTE: the legend handles should be colored squares, not line markers as we are making a bar plot
# legend_handles = [
#     plt.Line2D([0], [0],
#                color=MethodColorMap[method],
#                marker='s',
#                linestyle='',
#                markersize=10,
#                label=method.value,
#                alpha=MethodBarAlpha,
#                markeredgecolor='none', # remove the edge color
#                markeredgewidth=0,      # set edge width to 0 to reliably remove the edge lines
#     )
#     for method in MethodColorMap.keys()
# ]

all_plotted_methods = list(MethodColorMap.keys())
all_plotted_methods_order_dict = get_actual_method_orders(all_plotted_methods)
ordered_all_plotted_methods = list(all_plotted_methods_order_dict.keys())
legend_handles = [
    Patch(
        facecolor=MethodColorMap[method], # Use facecolor for the fill of the rectangle
        edgecolor='none',                 # Remove the edge line
        alpha=MethodBarAlpha,
        label=method.value
    )
    for method in ordered_all_plotted_methods
]

# add the legend to the figure, not to the axes
fig.legend(
    handles=legend_handles,
    loc='upper center',
    ncol=len(MethodColorMap),  # Number of columns in the legend
    # fontsize=10,  # Adjust font size for the legend
    frameon=False,  # Set frame around the legend
    # ncol=len(MethodBarOrders),
    # fontsize=12,  # Adjust font size for the legend
    # frameon=False,  # No frame around the legend
    # bbox_to_anchor=(0.5, 1.05)  # Position the legend above the subplots
    borderpad = 0.01,
)

# Fine-tune this with GUI
plt.subplots_adjust(
    top = 0.915,
    bottom = 0.060,
    left = 0.055,
    right = 0.985,
    hspace = 0.280,
    wspace = 0.140
)
        


# the bbox_inches='tight' option will help crop the figure to the content
plt.savefig(
    _THIS_SCRIPT_DIR/'fig-attention-throughput.pdf',
    bbox_inches='tight',
    pad_inches=0.01
)
# plt.show()
