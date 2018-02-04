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


def interp_target_rank_recall(cmc, target_rank):
    if(target_rank < cmc[0][0] or target_rank > cmc[0][-1]):
        print 'target_fpr out of bound, will return -1'
        return -1.0

    for i, rank in enumerate(cmc[0]):
        if rank > target_rank:
            break

    if cmc[0][i - 1] == target_rank:
        target_recall = cmc[1][i - 1]
    else:
        target_recall = bilinear_interp(target_rank,
                                        cmc[0][i - 1], cmc[0][i],
                                        cmc[1][i - 1], cmc[1][i]
                                        )

    return target_recall


def load_result_data(folder, probeset_name):
    #    n_distractors = generate_n_distractors()
    print '===> Load result data from ', folder

    all_files = os.listdir(folder)
#    print 'all_files: ', all_files
    cmc_files = sorted(
        [a for a in all_files if fnmatch(a, 'cmc*%s*_1.json' % probeset_name)])[::-1]
#    print 'cmc_files: ', cmc_files

    if not cmc_files:
        return None

    cmc_dict = {}
    for i, filename in enumerate(cmc_files):
        with open(os.path.join(folder, filename), 'r') as f:
            cmc_dict[n_distractors[i]] = json.load(f)

    rocs = []

    for i in n_distractors:
        rocs.append(cmc_dict[i]['roc'])

    cmcs = []

    for i in n_distractors:
        for j in range(len(cmc_dict[i]['cmc'][0])):
            cmc_dict[i]['cmc'][0][j] += 1

        cmcs.append(cmc_dict[i]['cmc'])

    rank_1 = [cmc_dict[n]['cmc'][1][0]
              for n in n_distractors]

    rank_10 = []
    for i in range(len(n_distractors)):
        target_recall = interp_target_rank_recall(cmcs[i], 10)
        rank_10.append(target_recall)

    roc_10K = cmc_dict[10000]['roc']
    roc_1M = cmc_dict[1000000]['roc']

    return {
        'rocs': rocs,
        'cmcs': cmcs,
        'rank_1': rank_1,
        'rank_10': rank_10,
        'roc_10k': roc_10K,
        'roc_1M': roc_1M
    }


# def load_your_result(your_result_dir, probeset_name, feat_ending=None):
#     '''
#     specify this function, if your result file names are not in the same format
#     as other methods.
#     '''

#     #    n_distractors = generate_n_distractors()
#     if '.' in feat_ending:
#         feat_ending = feat_ending.split('.')[0]

#     fn_tmpl = '%s_megaface' % probeset_name
#     if feat_ending:
#         fn_tmpl += feat_ending

#     fn_tmpl += '_{}_1.json'

#     cmc = osp.join(
#         your_result_dir, 'cmc_' + fn_tmpl)
# #        your_result_dir, 'cmc_megaface_{}_1.json')
#     cmc_files = [cmc.format(i) for i in n_distractors]

#     cmc_dict = {}
#     for i, filename in enumerate(cmc_files):
#         with open(filename, 'r') as f:
#             cmc_dict[n_distractors[i]] = json.load(f)

# #    matches = osp.join(
# #        your_result_dir, 'matches_' + fn_tmpl)
# # your_result_dir, 'matches_megaface_{}_1.json')
# #    matches_files = [
# #        matches.format(i) for i in n_distractors]
# #
# #    matches_dict = {}
# #    for i, filename in enumerate(matches_files):
# #        with open(filename, 'r') as f:
# #            matches_dict[n_distractors[i]] = json.load(f)
# #
#     rank_1 = [cmc_dict[n]['cmc'][1][0] for n in n_distractors]
#     rocs = [cmc_dict[n]['roc'] for n in n_distractors]

#     roc_10K = cmc_dict[10000]['roc']
#     roc_1M = cmc_dict[1000000]['roc']

#     # print cmc_dict[1000000]['cmc'][1][0]
#     # print cmc_dict[10]['roc']

#     return {'rank_1': rank_1,
#             'rocs': rocs,
#             'roc_10k': roc_10K,
#             'roc_1M': roc_1M
#             }


def calc_target_tpr_and_rank(rocs, rank_1, rank_10, save_dir, method_label=None):
    print '===> Calc and save TPR@FPR=1e-6 for method: ', method_label
    target_fpr = 1e-6
    fn_tpr = osp.join(save_dir, 'TPRs-at-FPR_%g' % target_fpr)
    fn_rank = osp.join(save_dir, 'rank_vs_distractors')

    if method_label:
        fn_tpr += '_' + method_label
        fn_rank += '_' + method_label
    else:
        method_label = "YOUR Method"

    fn_tpr += '.txt'
    fn_rank += '.txt'

    fp_tpr = open(fn_tpr, 'w')

    write_string = 'TPR@FPR=%g at different #distractors\n' % target_fpr
    write_string += '#distractors  TPR\n'
    print write_string
    fp_tpr.write(write_string)
    for i, roc in enumerate(rocs):
        target_tpr = interp_target_tpr(roc, target_fpr)
        write_string = '%7d %5.4f\n' % (n_distractors[i], target_tpr)
        print write_string
        fp_tpr.write(write_string)

    fp_tpr.close()

    print '===> Save Rank_1 under different #distractors for method: ', method_label
    fp_rank = open(fn_rank, 'w')
    write_string = 'Rank_1 recall at different #distractors\n'
    write_string += '#distractors  recall\n'
    print write_string
    fp_rank.write(write_string)

    for i, rank in enumerate(rank_1):
        write_string = '%7d  %5.4f\n' % (n_distractors[i], rank)
        print write_string
        fp_rank.write(write_string)

    write_string = '\nRank_10 recall at different #distractors\n'
    write_string += '#distractors  recall\n'
    print write_string
    fp_rank.write(write_string)

    for i, rank in enumerate(rank_10):
        write_string = '%7d  %5.4f\n' % (n_distractors[i], rank)
        print write_string
        fp_rank.write(write_string)

    fp_rank.close()


#%matplotlib inline
def plot_megaface_result(your_result_dir, your_method_label,
                         probeset_name,
                         other_methods_dir=None,
                         save_tpr_and_rank1_for_others=False):
    probeset_name = probeset_name.lower()
    if not probeset_name in ['facescrub', 'fgnet']:
        raise Exception(
            'probeset name must be either "facescrub" or "fgnet" !')

    save_dir = './rlt_%s_%s' % (probeset_name, your_method_label)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

#    n_distractors = generate_n_distractors()

    print 'n_distractors: ', n_distractors

    print '===> Loading data for probset {} from: {}'.format(probeset_name, your_result_dir)
    # your_result = load_your_result(your_result_dir, probeset_name, feat_ending)
    your_result = load_result_data(your_result_dir, probeset_name)
    rocs = your_result['rocs']
    cmcs = your_result['cmcs']
    rank_1 = your_result['rank_1']
    rank_10 = your_result['rank_1']

    calc_target_tpr_and_rank(rocs, rank_1, rank_10, save_dir)

    print '===> Plotting Verification ROC under different #distractors'
    fig = plt.figure(figsize=(16, 12), dpi=100)

    colors = ['g', 'r', 'b', 'c', 'm', 'y']
    labels = [str(it) for it in n_distractors]

    # plt.semilogx(rocs[0][0], rocs[0][1], 'g', label='10')
    # plt.semilogx(rocs[1][0], rocs[1][1], 'r', label='100')
    # plt.semilogx(rocs[2][0], rocs[2][1], 'b', label='1000')
    # plt.semilogx(your_result['roc_10k'][0],
    #              your_result['roc_10k'][1], 'c', label='10000')
    # plt.semilogx(rocs[4][0], rocs[4][1], 'm', label='100000')
    # plt.semilogx(your_result['roc_1M'][0],
    #              your_result['roc_1M'][1], 'y', label='1000000')

    for i in range(len(n_distractors)):
        plt.semilogx(rocs[i][0], rocs[i][1], colors[i], label=labels[i])

    plt.xlim([1e-8, 1])
    plt.ylim([0, 1])

    plt.grid()
    plt.legend(loc='lower right')
    plt.show()
    fig.savefig(osp.join(save_dir, 'roc_under_diff_distractors.png'),
                bbox_inches='tight')

    print '===> Plotting Identification CMC under different #distractors'
    fig = plt.figure(figsize=(16, 12), dpi=100)
    for i in range(len(n_distractors)):
        plt.semilogx(cmcs[i][0], cmcs[i][1], colors[i], label=labels[i])

    plt.xlim([1, 1e6])
    plt.ylim([0, 1])

    plt.grid()
    plt.legend(loc='lower right')
    plt.show()
    fig.savefig(osp.join(save_dir, 'cmc_under_diff_distractors.png'),
                bbox_inches='tight')

    print '===> Load result data for all the other methods'
    other_method_list = []
    other_methods_data = []

    if other_methods_dir:
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

    if other_method_list:
        other_methods_data = {}

        for method in other_method_list:
            result_data = load_result_data(
                os.path.join(other_methods_dir, method), probeset_name)

            if result_data is not None:
                other_methods_data[method] = load_result_data(
                    os.path.join(other_methods_dir, method), probeset_name)
        other_method_list = other_methods_data.keys()

        if save_tpr_and_rank1_for_others:
            for name in other_method_list:
                calc_target_tpr_and_rank(other_methods_data[name]['rocs'],
                                         other_methods_data[name]['rank_1'],
                                         other_methods_data[name]['rank_10'],
                                         save_dir, name)

    print '===> Plotting ROC under 10K distractors for your method'
    fig = plt.figure(figsize=(20, 10), dpi=200)
    ax = plt.subplot(111)
    ax.semilogx(your_result['roc_10k'][0],
                your_result['roc_10k'][1], label=your_method_label)

    if other_method_list:
        print '===> Plotting ROC under 10K distractors for all the other methods'

        for name in other_method_list:
            ax.semilogx(other_methods_data[name]['roc_10k'][0],
                        other_methods_data[name]['roc_10k'][1],
                        label=name,
                        c=np.random.rand(3))

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_xlim([1e-6, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.grid()
#    plt.legend()
    plt.show()
    fig.savefig(osp.join(save_dir, 'verification_roc_10K.png'),
                bbox_inches='tight')

    print '===> Plotting ROC under 1M distractors for your method'
    fig = plt.figure(figsize=(20, 10), dpi=200)
    ax = plt.subplot(111)
    ax.semilogx(your_result['roc_1M'][0],
                your_result['roc_1M'][1], label=your_method_label)

    if other_method_list:
        print '===> Plotting ROC under 1M distractors for all the other methods'

        for name in other_method_list:
            ax.semilogx(other_methods_data[name]['roc_1M'][0],
                        other_methods_data[name]['roc_1M'][1],
                        label=name,
                        c=np.random.rand(3))

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_xlim([1e-6, 1])
    ax.set_ylim([0, 1])

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.grid()
#    plt.legend()
    plt.show()
    fig.savefig(osp.join(save_dir, 'verification_roc_1M.png'),
                bbox_inches='tight')

    print '===> Plotting recall vs rank under 10K distractors for your method'
    fig = plt.figure(figsize=(20, 10), dpi=200)
    ax = plt.subplot(111)
    ax.semilogx(your_result['cmcs'][3][0],
                your_result['cmcs'][3][1], label=your_method_label)

    if other_method_list:
        print '===> Plotting recall vs rank under 10K distractors for all the other methods'

        for name in other_method_list:
            ax.semilogx(other_methods_data[name]['cmcs'][3][0],
                        other_methods_data[name]['cmcs'][3][1],
                        label=name,
                        c=np.random.rand(3))

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_xlim([1, 1e4])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Rank')
    ax.set_ylabel('Identification Rate (Recall)')

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.grid()
#    plt.legend()
    plt.show()
    fig.savefig(osp.join(save_dir, 'identification_recall_vs_rank_10K.png'),
                bbox_inches='tight')

    print '===> Plotting recall vs rank under 1M distractors for your method'
    fig = plt.figure(figsize=(20, 10), dpi=200)
    ax = plt.subplot(111)
    ax.semilogx(your_result['cmcs'][-1][0],
                your_result['cmcs'][-1][1], label=your_method_label)

    if other_method_list:
        print '===> Plotting recall vs rank under 1M distractors for all the other methods'

        for name in other_method_list:
            ax.semilogx(other_methods_data[name]['cmcs'][-1][0],
                        other_methods_data[name]['cmcs'][-1][1],
                        label=name,
                        c=np.random.rand(3))

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_xlim([1, 1e6])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Rank')
    ax.set_ylabel('Identification Rate (Recall)')

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.grid()
#    plt.legend()
    plt.show()
    fig.savefig(osp.join(save_dir, 'identification_recall_vs_rank_1M.png'),
                bbox_inches='tight')


    # if other_method_list:
    #     print '===> Plotting rank_1 vs #distractors for all the other methods'
    #     fig = plt.figure(figsize=(10, 10), dpi=200)
    #     dd = [plt.semilogx(n_distractors, other_methods_data[name]['rank_1'],
    #                     label=name) for name in other_method_list]
    #     # dd = [plt.semilogx(n_distractors, other_methods_data[name]['rank_1'],
    #     #                    label=name) for name in other_method_list]
    #     dd += [plt.semilogx(n_distractors, rank_1, label=your_method_label)]
    #     plt.xscale('log')
    #     plt.grid()
    #     plt.legend()
    #     plt.show()
    #     fig.savefig(osp.join(save_dir, 'identification_rank_1_vs_distractors_small.png'),
    #                 bbox_inches='tight')

    print '===> Plotting rank_1 vs #distractors for your method'
    fig = plt.figure(figsize=(20, 10), dpi=100)
    ax = plt.subplot(111)
    ax.semilogx(n_distractors, rank_1, label=your_method_label)

    if other_method_list:
        print '===> Plotting rank_1 vs #distractors for all the other methods'

        for name in other_method_list:
            ax.semilogx(
                n_distractors,
                other_methods_data[name]['rank_1'],
                label=name,
                c=np.random.rand(3))

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_xlabel('# distractors (logscale)')
    ax.set_ylabel('Identification rate')

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.grid()
#    plt.legend()
    plt.show()
    fig.savefig(osp.join(save_dir, 'identification_rank_1_vs_distractors.png'),
                bbox_inches='tight')


if __name__ == '__main__':
    your_result_dir = r'C:\zyf\dataset\megaface\Challenge1External\facenet'
    your_method_label = 'facenet'

    probesets = ['facescrub', 'fgnet']
    # feat_ending = '_feat'

    # other_methods_dir = None
    other_methods_dir = r'C:\zyf\dataset\megaface\Challenge1External'
    save_tpr_and_rank1_for_others = False
    # save_tpr_and_rank1_for_others = True

    for probeset_name in probesets:
        plot_megaface_result(your_result_dir, your_method_label,
                             probeset_name,
                             other_methods_dir,
                             save_tpr_and_rank1_for_others
                             )
