#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 19:23:16 2023

@author: ikalvet
"""
import pyrosetta as pyr
import pyrosetta.rosetta
import os, sys
import pandas as pd

SCRIPT_PATH = os.path.dirname(__file__)
sys.path.append(f"{SCRIPT_PATH}/../../utils")
import no_ligand_repack
import scoring_utils
import design_utils


comparisons = {'<=': '__le__',
               '<': '__lt__',
               '>': '__gt__',
               '>=': '__ge__',
               '=': '__eq__'}


def fix_scorefxn(sfxn, allow_double_bb=False):
    opts = sfxn.energy_method_options()
    opts.hbond_options().decompose_bb_hb_into_pair_energies(True)
    opts.hbond_options().bb_donor_acceptor_check(not allow_double_bb)
    sfxn.set_energy_method_options(opts)


def score_design(pose, sfx, catres):
    df_scores = pd.DataFrame()

    # Adding Rosetta scores to df
    sfx(pose)
    for k in pose.scores:
        df_scores.at[0, k] = pose.scores[k]

    df_scores.at[0, "score_per_res"] = df_scores.at[0, "total_score"]/pose.size()

    return df_scores


def filter_scores(scores):
    """
    Filters are defined in this importable module
    """
    filtered_scores = scores.copy()

    for s in filters.keys():
        if filters[s] is not None and s in scores.keys():
            val = filters[s][0]
            sign = comparisons[filters[s][1]]
            filtered_scores =\
              filtered_scores.loc[(filtered_scores[s].__getattribute__(sign)(val))]
            n_passed = len(scores.loc[(scores[s].__getattribute__(sign)(val))])
            print(f"{s:<24} {filters[s][1]:<2} {val:>7.3f}: {len(filtered_scores)} "
                  f"designs left. {n_passed} pass ({(n_passed/len(scores))*100:.0f}%).")
    return filtered_scores


filters = {
           }
