# -*- coding: utf-8 -*-
"""
Created on Sat Feb 03 06:44:14 2018

@author: zhaoy
"""
import _init_paths
from plot_megaface_result import plot_megaface_result


if __name__ == '__main__':
    my_method_dirs = [
        r'C:\zyf\dnn_models\face_models\face_eval_idcard1M_eval\eval-results\sphereface-64-1220',
        r'C:\zyf\dnn_models\face_models\face_eval_idcard1M_eval\eval-results\sphereface-64-ms-fixed2-refined-0128'
    ]

    my_method_labels = [
        '1220-MS1M-WX-part',
        '0128-MS1M-WX-2refine'
    ]

    probe = 'idProbe'
    # feat_ending = '_feat'

    other_methods_dir = None
    # other_methods_dir = r'C:\zyf\dataset\megaface\Challenge1External'
    save_tpr_and_rank1_for_others = False
    # save_tpr_and_rank1_for_others = True

    save_dir = './rlt_idcard1M_%s_facex_sphere64' % probe
    plot_megaface_result(my_method_dirs, my_method_labels,
                            probe,
                            save_dir,
                            other_methods_dir,
                            save_tpr_and_rank1_for_others
                            )
