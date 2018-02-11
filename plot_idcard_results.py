# -*- coding: utf-8 -*-
"""
Created on Sat Feb 03 06:44:14 2018

@author: zhaoy
"""
from plot_megaface_result import plot_megaface_result


if __name__ == '__main__':
    # change this path to your results' real path
    my_result_dirs = [
        r'devkit/eval-results-idcard1M'
    ]

    my_method_labels = [
        'FaceX-ZYF-1120'
    ]

    # probeset_name = 'facescrub'
    probesets = ['idProbe']
    # feat_ending = '_feat'

    other_methods_dir = None
    save_tpr_and_rank1_for_others = False
    # save_tpr_and_rank1_for_others = True

    for i, my_dir in enumerate(my_result_dirs):
        plot_megaface_result(my_dir, my_method_labels[i],
                    probeset_name,
                    other_methods_dir,
                    save_tpr_and_rank1_for_others
                    )
