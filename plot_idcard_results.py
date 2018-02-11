# -*- coding: utf-8 -*-
"""
Created on Sat Feb 03 06:44:14 2018

@author: zhaoy
"""
import sys
from plot_megaface_result import plot_megaface_result


if __name__ == '__main__':
    # change this path to your results' real path
    my_result_dir = r'devkit/eval-results-idcard1M'
    my_method_label = 'FaceX'

    if len(sys.argv) > 1:
        my_result_dir = sys.argv[1]
    if len(sys.argv) > 2:
        my_method_label = sys.argv[2]

    probeset_name = 'idProbe'

    # feat_ending = '_feat'

    other_methods_dir = None
    save_tpr_and_rank1_for_others = False
    # save_tpr_and_rank1_for_others = True

    plot_megaface_result(my_result_dir, my_method_label,
                         probeset_name,
                         other_methods_dir,
                         save_tpr_and_rank1_for_others
                         )
