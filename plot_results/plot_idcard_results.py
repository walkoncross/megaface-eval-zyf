# -*- coding: utf-8 -*-
"""
Created on Sat Feb 03 06:44:14 2018

@author: zhaoy
"""
import sys
from plot_megaface_result import plot_megaface_result


if __name__ == '__main__':
    your_result_dir = r'C:\zyf\dnn_models\face_models\face_eval_idcard1M_eval\eval-results\sphereface-64-1220'
    your_method_label = 'facex-1220'

    probesets = ['idProbe']
    # feat_ending = '_feat'

#    other_methods_dir = None
    other_methods_dir = r'C:\zyf\dnn_models\face_models\face_eval_idcard1M_eval\eval-results'
    # save_tpr_and_rank1_for_others = False
    save_tpr_and_rank1_for_others = True

    for probeset_name in probesets:
        plot_megaface_result(your_result_dir, your_method_label,
                             probeset_name,
                             other_methods_dir,
                             save_tpr_and_rank1_for_others
                             )
