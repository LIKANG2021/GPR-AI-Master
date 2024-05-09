# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 11:49:59 2023

@author: PC
"""
#确定画图变量和python库
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def remove_jagged_edges(data, window_size=3):
    if window_size % 2 == 0:
        window_size += 1  # 窗口大小应为奇数

    half_window = window_size // 2
    smoothed_data = data.copy()

    for i in range(half_window, len(data) - half_window):
        smoothed_data[i] = np.mean(data[i - half_window: i + half_window + 1])

    return smoothed_data

def huatu_cls(GPR_Frequency, Time_window, Theoretical_Shield_tail_Gap, Excelent_metric, Qualified_metric, Insufficient_metric, Inputfile_Name, ML_output, jpg_name):
    Data_input = pd.read_csv(Inputfile_Name, delimiter=',', header=None)

    ML_output = remove_jagged_edges(ML_output)

    # 画图代码
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.rcParams['figure.autolayout'] = True
    config = {"font.family": 'serif', "font.size": 10, "mathtext.fontset": 'stix', "font.serif": ['Times New Roman'],
              'figure.figsize': (0.393701 * 8, 0.393701 * 6)}
    plt.rcParams.update(config)
    # 设置图的整体大小和宽度
    fig_width = 30  # 图的宽度为16cm
    fig_height = 10  # 图的高度为8cm

    # 画图主要的变量信息，都是根据传入变量变换而来的
    Inputfile_Basename, Inputfile_extension = os.path.splitext(Inputfile_Name)
    Inputfile_Basename.split("_")
    Monit_Line_No = Inputfile_Basename.split("_")[0]  # 字符串
    Segment_Tickness = float(Inputfile_Basename.split("_")[1])  # 浮点数，2位小数
    Monit_Line_Len = float(Inputfile_Basename.split("_")[2])  # 浮点数，2位小数
    y_segment_top = Segment_Tickness
    y_grouting_top = 0.35 + ML_output.flatten()
    y_soil_top = 1
    x_max = ML_output.shape[0]
    y_max = 0.75
    x = [i for i in range(x_max)]
    # 创建一个包含三个子图的画布
    fig, axes = plt.subplots(1, 3, figsize=(fig_width / 2.54, fig_height / 2.54), dpi=600)

    # 左图
    im = axes[0].imshow(Data_input
                        , aspect='auto'
                        , cmap='jet'
                        )
    axes[0].set_title("Original GPR Data of Line " + os.path.basename(str(Monit_Line_No))
                      , fontweight='bold')
    axes[0].set_xlabel("Monitoring Line/m")
    axes[0].set_ylabel("Time Window/ns")

    axes[0].locator_params(axis='y', nbins=7)

    y_ticks = np.linspace(0, Data_input.shape[0], int(Time_window / 10 + 1))
    new_yticklabels = [str(i * 10) for i in range(int(Time_window / 10 + 1))]
    axes[0].set_yticks(y_ticks)
    axes[0].set_yticklabels(new_yticklabels)

    x_ticks = np.linspace(0, Data_input.shape[1], 5)
    new_xticklabels = [f"{i:.2f}" for i in np.linspace(0, Monit_Line_Len, 5)]
    axes[0].set_xticks(x_ticks)
    axes[0].set_xticklabels(new_xticklabels)

    # 中图
    # 盾尾间隙线
    colors_model = ["gray", "#1DDFC0", "#FFA73F"]
    segment_clor = colors_model[0]
    grouting_clor = colors_model[1]
    soil_clor = colors_model[2]
    axes[1].axhline(y_segment_top + Theoretical_Shield_tail_Gap
                    , color='r'
                    , ls='-.'
                    , lw=0.5
                    )
    axes[1].fill_between(x
                         , 0
                         , y_segment_top
                         , color=segment_clor
                         , alpha=0.5
                         )
    axes[1].plot(x
                 , y_grouting_top
                 , color=grouting_clor
                 )
    axes[1].fill_between(x
                         , Segment_Tickness
                         , y_grouting_top
                         , color=grouting_clor
                         , alpha=1
                         )

    axes[1].fill_between(x
                         , y_grouting_top
                         , y_max
                         , color=soil_clor
                         , alpha=1
                         )
    axes[1].set_title("Thickness Distribution"
                      , fontweight='bold')
    axes[1].set_xlim(0, x_max)
    axes[1].set_ylim(0, y_max)
    axes[1].set_xlabel("Monitoring Line/m")
    axes[1].set_ylabel("Thickness/m")

    legend_labels = {
        segment_clor: 'Segment',
        grouting_clor: 'Grout',
        soil_clor: 'Soil'
    }
    legend_handles = [plt.Rectangle((0, 0), 0.5, 0.5, color=color, label=legend_labels[color]) for color in
                      set(colors_model)]
    axes[1].legend(handles=legend_handles
                   , ncol=3
                   , loc='lower center'
                   , fontsize=8
                   , bbox_to_anchor=(0.5, -0.3)
                   , frameon=False
                   , handlelength=1
                   , handleheight=1)

    x_ticks = np.linspace(0, x_max, 5)
    new_xticklabels = [f"{i:.2f}" for i in np.linspace(0, Monit_Line_Len, 5)]
    axes[1].set_xticks(x_ticks)
    axes[1].set_xticklabels(new_xticklabels)

    # 注浆厚度评价图
    # 评价结果颜色划分
    ratio_percent = (ML_output.flatten() / Theoretical_Shield_tail_Gap) * 100
    colors = []
    Excelent_color = "#00E7B7"
    Qualified_color = "#FFA73F"
    Insufficient_color = "#FF002D"
    for a in ratio_percent:
        if a > Excelent_metric:
            colors.append(Excelent_color)
        elif Qualified_metric < a <= Excelent_metric:
            colors.append(Qualified_color)
        else:
            colors.append(Insufficient_color)
    for i in range(x_max):
        axes[2].bar(i
                    , (ML_output.flatten()[i]) * 100 / Theoretical_Shield_tail_Gap
                    , color=colors[i]
                    , width=1)

    # axes[2].scatter(x
    #              ,ML_output.values.flatten()
    #              ,color=colors
    #              ,s=2
    #             )
    axes[2].set_title("Thickness Evaluation"
                      , fontweight='bold')
    axes[2].set_xlim(0, x_max)
    axes[2].set_xlabel("Monitoring Line/m")
    axes[2].set_ylabel("Thickness/Shield Gap (%)")
    legend_labels = {
        Excelent_color: 'Excellent',
        Qualified_color: 'Qualified',
        Insufficient_color: 'Insufficient'
    }
    legend_handles = [plt.Rectangle((0, 0), 0.5, 0.5, color=color, label=legend_labels[color]) for color in set(colors)]
    axes[2].legend(handles=legend_handles
                   , ncol=3
                   , loc='lower center'
                   , fontsize=8
                   , bbox_to_anchor=(0.5, -0.3)
                   , frameon=False
                   , columnspacing=0.5)
    x_ticks = np.linspace(0, x_max, 5)
    new_xticklabels = [f"{i:.2f}" for i in np.linspace(0, Monit_Line_Len, 5)]
    axes[2].set_xticks(x_ticks)
    axes[2].set_xticklabels(new_xticklabels)

    plt.savefig(jpg_name
                , dpi=700
                , bbox_inches='tight'
                , pad_inches=0.01)
