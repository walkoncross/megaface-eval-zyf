# -*- coding: utf-8 -*-
"""
Created on Thu Feb 01 08:07:14 2018

@author: zhaoy
"""
import os
import os.path as osp

import numpy as np
import json

from fnmatch import fnmatch

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt

from interpolation import bilinear_interp


def generate_n_distractors():
    n_distractors = [10 ** i for i in range(1, 7)]

    return n_distractors


n_distractors = generate_n_distractors()


def get_result_for_other_method(folder, probeset_name):
    #    n_distractors = generate_n_distractors()

    all_files = os.listdir(folder)
#    print 'all_files: ', all_files
    cmc_files = sorted(
        [a for a in all_files if fnmatch(a, 'cmc*%s*_1.json' % probeset_name)])[::-1]
#    print 'cmc_files: ', cmc_files

    cmc_dict = {}
    for i, filename in enumerate(cmc_files):
        with open(os.path.join(folder, filename), 'r') as f:
            cmc_dict[n_distractors[i]] = json.load(f)

    rank_1 = [cmc_dict[n]['cmc'][1][0]
              for n in n_distractors]
    roc_10K = cmc_dict[10000]['roc']
    roc_1M = cmc_dict[1000000]['roc']

    return {'rank_1': rank_1,
            'roc_10k': roc_10K,
            'roc_1M': roc_1M
            }


def load_result(result_root_dir, probeset_name, feat_ending=None):
    #    n_distractors = generate_n_distractors()
    if '.' in feat_ending:
        feat_ending = feat_ending.split('.')[0]

    fn_tmpl = '%s_megaface' % probeset_name
    if feat_ending:
        fn_tmpl += feat_ending

    fn_tmpl += '_{}_1.json'


    cmc = osp.join(
        result_root_dir, 'cmc_' + fn_tmpl)
#        result_root_dir, 'cmc_megaface_{}_1.json')
    cmc_files = [cmc.format(i) for i in n_distractors]

    cmc_dict = {}
    for i, filename in enumerate(cmc_files):
        with open(filename, 'r') as f:
            cmc_dict[n_distractors[i]] = json.load(f)

#    matches = osp.join(
#        result_root_dir, 'matches_' + fn_tmpl)
# result_root_dir, 'matches_megaface_{}_1.json')
#    matches_files = [
#        matches.format(i) for i in n_distractors]
#
#    matches_dict = {}
#    for i, filename in enumerate(matches_files):
#        with open(filename, 'r') as f:
#            matches_dict[n_distractors[i]] = json.load(f)
#
    rank_1 = [cmc_dict[n]['cmc'][1][0] for n in n_distractors]
    rocs = [cmc_dict[n]['roc'] for n in n_distractors]

    # print cmc_dict[1000000]['cmc'][1][0]
    # print cmc_dict[10]['roc']

    return {'rank_1': rank_1,
            'rocs': rocs
            }


def interp_target_tpr(roc, target_fpr):
    if(target_fpr < roc[0][0] or target_fpr > roc[0][-1]):
        print 'target_fpr out of bound, will return -1'
        return -1.0

    for i, fpr in enumerate(roc[0]):
        if fpr > target_fpr:
            break

    target_tpr = bilinear_interp(target_fpr,
                                 roc[0][i - 1], roc[0][i],
                                 roc[1][i - 1], roc[1][i]
                                 )

    return target_tpr


#%matplotlib inline
def main(result_root_dir, method_label,
         probeset_name, feat_ending,
         other_methods_dir):
    probeset_name = probeset_name.lower()
    if not probeset_name in ['facescrub', 'fgnet']:
        raise Exception(
            'probeset name must be either "facescrub" or "fgnet" !')

    save_dir = './rlt_%s_%s' % (probeset_name, method_label)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

#    n_distractors = generate_n_distractors()

    print 'n_distractors: ', n_distractors

    print '===> Loading data for probset {} from: {}'.format(probeset_name, result_root_dir)
    result = load_result(result_root_dir, probeset_name, feat_ending)
    rocs = result['rocs']
    rank_1 = result['rank_1']

    print '===> Calc and save TPR@FPR=1e-6'
    target_fpr = 1e-6
    fp = open(osp.join(save_dir, 'TPRs-at-FPR_%g.txt' % target_fpr), 'w')
    for i, roc in enumerate(rocs):
        target_tpr = interp_target_tpr(roc, target_fpr)
        write_string = 'TPR@FPR=%g (with %7d distractors): %5.4f\n' % (target_fpr,
                           n_distractors[i], target_tpr)
        print write_string

        fp.write(write_string)
    fp.close()

    print '===> Save Rank_1 under different #distractors'
    fp = open(osp.join(save_dir, 'Rank_1_vs_distractors.txt'), 'w')
    for i, rank in enumerate(rank_1):
        write_string = 'Rank_1 (with %7d distractors): %5.4f\n' % (
                            n_distractors[i], rank)
        print write_string

        fp.write(write_string)
    fp.close()


    print '===> Plotting Verification ROC under different #distractors'
    fig = plt.figure(figsize=(12, 9), dpi=100)

    plt.semilogx(rocs[0][0], rocs[0][1], 'g', label='10')
    plt.semilogx(rocs[1][0], rocs[1][1], 'r', label='100')
    plt.semilogx(rocs[2][0], rocs[2][1], 'b', label='1000')
    plt.semilogx(rocs[3][0], rocs[3][1], 'c', label='10000')
    plt.semilogx(rocs[4][0], rocs[4][1], 'm', label='100000')
    plt.semilogx(rocs[5][0], rocs[5][1], 'y', label='1000000')
    plt.grid()
    plt.legend()
    plt.show()
    fig.savefig(osp.join(save_dir, 'roc_under_diff_distractors.png'),
                bbox_inches='tight')

    # plot other methods
    other_method_list = os.listdir(other_methods_dir)
    print 'other_method_list: ', other_method_list

    # ['3divi',
    #  'deepsense',
    #  'ntech',
    #  'faceall_norm',
    #  'Vocord',
    #  'Barebones_FR',
    #  'ntech_small',
    #  'deepsense_small',
    #  'SIAT_MMLAB',
    #  'faceall',
    #  'facenet',
    #  'ShanghaiTech']

    print '===> Get result data for all the other methods'
    data = {method: get_result_for_other_method(os.path.join(other_methods_dir, method), probeset_name)
            for method in other_method_list}

    print '===> Plotting ROC under 10K distractors for all the methods'
    fig = plt.figure(figsize=(20, 10), dpi=200)
    ax = plt.subplot(111)

    for name in other_method_list:
        ax.semilogx(data[name]['roc_10k'][0],
                    data[name]['roc_10k'][1],
                    label=name,
                    c=np.random.rand(3))

    ax.semilogx(rocs[3][0], rocs[3][1], label=method_label)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_xlim([1e-6, 1])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.grid()
    plt.legend()
    plt.show()
    fig.savefig(osp.join(save_dir, 'verification_roc_10K.png'),
                bbox_inches='tight')

    print '===> Plotting ROC under 1M distractors for all the methods'
    fig = plt.figure(figsize=(20, 10), dpi=200)
    ax = plt.subplot(111)

    for name in other_method_list:
        ax.semilogx(data[name]['roc_1M'][0],
                    data[name]['roc_1M'][1],
                    label=name,
                    c=np.random.rand(3))
    ax.semilogx(rocs[5][0], rocs[5][1], label=method_label)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_xlim([1e-6, 1])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.grid()
    plt.legend()
    plt.show()
    fig.savefig(osp.join(save_dir, 'verification_roc_1M.png'),
                bbox_inches='tight')

    print '===> Plotting rank_1 vs #distractors for all the methods'
    fig = plt.figure(figsize=(10, 10), dpi=200)
    dd = [plt.semilogx(n_distractors, data[name]['rank_1'],
                       label=name) for name in other_method_list]
    # dd = [plt.semilogx(n_distractors, data[name]['rank_1'],
    #                    label=name) for name in other_method_list]
    dd += [plt.semilogx(n_distractors, rank_1, label=method_label)]
    plt.xscale('log')
    plt.grid()
    plt.legend()
    plt.show()
    fig.savefig(osp.join(save_dir, 'identification_rank_1_vs_distractors_small.png'),
                bbox_inches='tight')

    print '===> Plotting rank_1 vs #distractors for all the methods'
    fig = plt.figure(figsize=(20, 10), dpi=100)
    ax = plt.subplot(111)

    for name in other_method_list:
        ax.semilogx(
            n_distractors,
            data[name]['rank_1'],
            label=name,
            c=np.random.rand(3))

    ax.semilogx(n_distractors, rank_1, label=method_label)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_xlabel('# distractors (logscale)')
    ax.set_ylabel('Identification rate')

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.grid()
    plt.legend()
    plt.show()
    fig.savefig(osp.join(save_dir, 'identification_rank_1_vs_distractors.png'),
                bbox_inches='tight')


if __name__ == '__main__':
#    result_root_dir = r'C:\zyf\dnn_models\face_models\megaface-eval\eval-results\sphereface-64-1220'
#    method_label = 'FaceX-ZYF-1220'

#    result_root_dir = r'C:\zyf\dnn_models\face_models\megaface-eval\eval-results\sphereface-64-ms-fixed2-0107'
#    method_label = 'FaceX-ZYF-0107'

#    result_root_dir = r'C:\zyf\dnn_models\face_models\megaface-eval\eval-results\sphereface-64-ms-fixed2-refined-0128'
#    method_label = 'FaceX-ZYF-0128'

    result_root_dir = r'C:\zyf\dnn_models\face_models\megaface-eval\eval-results\sphereface-64-ms-merged-0131'
    method_label = 'FaceX-ZYF-0131'

    probeset_name = 'facescrub'
    feat_ending = '_feat'

    other_methods_dir = r'C:\zyf\dataset\megaface\Challenge1External'
    main(result_root_dir, method_label, probeset_name, feat_ending, other_methods_dir)
