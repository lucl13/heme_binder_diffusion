import sys, os, glob, shutil, subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pyrosetta as pyr
import pyrosetta.rosetta
import pyrosetta.distributed.io
import pyrosetta.rosetta.core.select.residue_selector as residue_selector
import json
import getpass
import argparse
import random
import copy
import time
import scipy.spatial
import io

import setup_fixed_positions_around_target

# MPNN scripts
SCRIPT_PATH = os.path.dirname(__file__)
sys.path.append(f"{SCRIPT_PATH}/../../lib/LigandMPNN")
import mpnn_api
from mpnn_api import MPNNRunner

# Utility scripts
sys.path.append(f"{SCRIPT_PATH}/../utils")
import no_ligand_repack
import scoring_utils
import design_utils



def get_crude_fastrelax(fastrelax):
    """
    Modifies your fastrelax method to run a very crude relax script
    MonomerRelax2019:
        protocols.relax.RelaxScriptManager: coord_cst_weight 1.0
        protocols.relax.RelaxScriptManager: scale:fa_rep 0.040
        protocols.relax.RelaxScriptManager: repack
        protocols.relax.RelaxScriptManager: scale:fa_rep 0.051
        protocols.relax.RelaxScriptManager: min 0.01
        protocols.relax.RelaxScriptManager: coord_cst_weight 0.5
        protocols.relax.RelaxScriptManager: scale:fa_rep 0.265
        protocols.relax.RelaxScriptManager: repack
        protocols.relax.RelaxScriptManager: scale:fa_rep 0.280
        protocols.relax.RelaxScriptManager: min 0.01
        protocols.relax.RelaxScriptManager: coord_cst_weight 0.0
        protocols.relax.RelaxScriptManager: scale:fa_rep 0.559
        protocols.relax.RelaxScriptManager: repack
        protocols.relax.RelaxScriptManager: scale:fa_rep 0.581
        protocols.relax.RelaxScriptManager: min 0.01
        protocols.relax.RelaxScriptManager: coord_cst_weight 0.0
        protocols.relax.RelaxScriptManager: scale:fa_rep 1
        protocols.relax.RelaxScriptManager: repack
        protocols.relax.RelaxScriptManager: min 0.00001
    """
    _fr = fastrelax.clone()
    script = ["coord_cst_weight 1.0",
              "scale:fa_rep 0.1",
              "repack",
              "coord_cst_weight 0.5",
              "scale:fa_rep 0.280",
              "repack",
              "min 0.01",
              "coord_cst_weight 0.0",
              "scale:fa_rep 1",
              "repack",
              "min 0.005",
              "accept_to_best"]
    filelines = pyrosetta.rosetta.std.vector_std_string()
    [filelines.append(l.rstrip()) for l in script]
    _fr.set_script_from_lines(filelines)
    return _fr


def setup_fastrelax(sfx, crude=False):
    fastRelax = pyrosetta.rosetta.protocols.relax.FastRelax(sfx, 1)
    if crude is True:
        fastRelax = get_crude_fastrelax(fastRelax)
    fastRelax.constrain_relax_to_start_coords(True)

    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides())
    e = pyrosetta.rosetta.core.pack.task.operation.ExtraRotamersGeneric()
    e.ex1(True)
    e.ex1aro(True)
    if crude is False:
        e.ex2(True)
        # e.ex1_sample_level(pyrosetta.rosetta.core.pack.task.ExtraRotSample(1))
    tf.push_back(e)
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.RestrictToRepacking())
    fastRelax.set_task_factory(tf)
    return fastRelax



parser = argparse.ArgumentParser()
parser.add_argument("--pdb", required=True, type=str, help="Input PDB")
parser.add_argument("--params", nargs="+", type=str, help="Params files")
parser.add_argument("--scoring", type=str, required=True, help="Path to a script that implement scoring methods for a particular design job.\n"
                    "Script must implement methods score_design(pose, sfx, catres) and filter_scores(scores), and a dictionary `filters` with filtering criteria.")
parser.add_argument("--nstruct", type=int, default=1, help="How many design iterations?")
parser.add_argument("--debug", action="store_true", default=False, help="For debugging.. terminates only and allows poking things interactively")
parser.add_argument("--align_atoms", nargs="+", type=str, help="Ligand atom names used for aligning the rotamers. Can also be proved with the scoring script.")
parser.add_argument("--keep_native", nargs="+", type=str, help="Residue positions that should not be redesigned. Use 'trb' as argument value to indicate that fixed positions should be taken from the 'con_hal_idx0' list in the corresponding TRB file provided with --trb flag.")
parser.add_argument("--trb", type=str, help="TRB file associated with the input scaffold. Required only when using --keep_native flag.")
parser.add_argument("--design_full", action="store_true", default=False, help="All positions are set designable. Apart from catalytic residues and those provided with --keep_native")
parser.add_argument("--iterate", action="store_true", default=False, help="runs mpnn/FastRelax iteratively")

args = parser.parse_args()

INPUT_PDB = args.pdb
params = args.params
scorefilename = "scorefile.txt"



## Loading the user-provided scoring module
sys.path.append(os.path.dirname(args.scoring))
scoring = __import__(os.path.basename(args.scoring.replace(".py", "")))
assert hasattr(scoring, "score_design")
assert hasattr(scoring, "filter_scores")
assert hasattr(scoring, "filters")


if args.keep_native is not None:
    assert args.trb is not None, "Must provide TRB file when using --keep_native argument"


"""
Getting PyRosetta started
"""
extra_res_fa = ""
if args.params is not None:
    extra_res_fa = "-extra_res_fa"
    for p in args.params:
        extra_res_fa += f" {p}"

NPROC = os.cpu_count()
if "SLURM_CPUS_ON_NODE" in os.environ:
    NPROC = os.environ["SLURM_CPUS_ON_NODE"]
elif "OMP_NUM_THREADS" in os.environ:
    NPROC = os.environ["OMP_NUM_THREADS"]


DAB = f"{SCRIPT_PATH}/../utils/DAlphaBall.gcc" # This binary was compiled on UW systems. It may or may not work correctly on yours
assert os.path.exists(DAB), "Please compile DAlphaBall.gcc and manually provide a path to it in this script under the variable `DAB`\n"\
                        "For more info on DAlphaBall, visit: https://www.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Filters/HolesFilter"


pyr.init(f"{extra_res_fa} -dalphaball {DAB} -beta_nov16 -run:preserve_header -mute all -gen_potential "
         f"-multithreading true -multithreading:total_threads {NPROC} -multithreading:interaction_graph_threads {NPROC}")

sfx = pyr.get_fa_scorefxn()
fastRelax = setup_fastrelax(sfx, crude=True)
fastRelax_proper = setup_fastrelax(sfx, crude=False)

###############################################
### PARSING PDB AND FINDING POCKET RESIDUES ###
###############################################
pdb_name = os.path.basename(INPUT_PDB).replace(".pdb", "")

input_pose = pyrosetta.pose_from_file(INPUT_PDB)
pose = input_pose.clone()


keep_native = []
if args.keep_native is not None:
    _trb = np.load(args.trb, allow_pickle=True)
    if args.keep_native[0] != "trb":
        poslist = _trb["con_hal_idx0"]
        ref_pdb = _trb["con_ref_pdb_idx"]
    
        native_positions = []
        accepted = []  # list of tuples with fixed positions [(A, ##),]
        for p in args.keep_native:
            if p.isnumeric():
                native_positions.append(p)
                accepted.append(("A", int(p)))
            elif not p.isnumeric() and "-" not in p:
                accepted.append((p[0], int(p[1:])))
            elif "-" in p:
                if p[0].isnumeric():
                    _ch = "A"
                    _rng = (int(p.split("-")[0]), int(p.split("-")[-1])+1)
                else:
                    _ch = p[0]
                    _rng = (int(p.split("-")[0][1:]), int(p.split("-")[-1])+1)
                for _n in range(_rng[0], _rng[1]):
                    native_positions.append(_n)
                    accepted.append((_ch, _n))
            else:
                print(f"Invalid value for -keep_native: {p}")
    
        acc = [i for i,p in enumerate(ref_pdb) if p in accepted]  # List of fixed position id's in the reference PDB list
        keep_native = [int(poslist[x])+1 for x in acc]  # Residue numbers of fixed positions in inpaint output
        keep_native = [i for i in keep_native if _trb["inpaint_seq"][i-1] == True and _trb["inpaint_str"][i-1] == True]
    elif args.keep_native[0] == "trb":
        poslist = _trb["con_hal_idx0"]
        ref_pdb = _trb["con_ref_pdb_idx"]
        keep_native = [int(x+1) for x in poslist]  # Residue numbers of fixed positions in inpaint output
        keep_native = [i for i in keep_native if _trb["inpaint_seq"][i-1] == True and _trb["inpaint_str"][i-1] == True]


print("Setting up MPNN API")
mpnnrunner = MPNNRunner(model_type="soluble_mpnn", ligand_mpnn_use_side_chain_context=False)  # starting with default checkpoint

for N in range(args.nstruct):
    iter_start_time = time.time()
    output_name = f"{pdb_name}_mDE_{N}"
    if os.path.exists(output_name + ".pdb"):
        print(f"Design {output_name} already exists, skipping iteration...")
        continue

    if args.debug is True:
        sys.exit(0)

    filt_scores = []
    N_iter = 0

    #########################################################
    ### Running MPNN until relaxed pose meets all filters ###
    #########################################################
    _pose2 = pose.clone()

    while len(filt_scores) == 0:
        if N_iter == 5:
            break

        if args.iterate is False:
            # Using the same input pose for each iteration
            _pose2 = pose.clone()

        pdbstr = pyrosetta.distributed.io.to_pdbstring(_pose2)

        print("Identifying pocket positions")
        # Re-evaluating the set of fixed residues at each iteration
        fixed_residues = []
        for rn in list(set(keep_native)):
            fixed_residues.append(_pose2.pdb_info().chain(rn)+str(_pose2.pdb_info().number(rn)))

        # Setting up MPNN runner
        inp = mpnnrunner.MPNN_Input()
        inp.pdb = pdbstr
        inp.fixed_residues = fixed_residues
        inp.temperature = 0.2
        inp.omit_AA = "CM"
        inp.batch_size = 5
        inp.number_of_batches = 1
        if N_iter == 0:
            inp.number_of_batches = 2

        print(f"Generating {inp.batch_size*inp.number_of_batches} initial guess sequences with ligandMPNN")
        mpnn_out = mpnnrunner.run(inp)


        ##############################################################################
        ### Finding which of the MPNN-packed structures has the best Rosetta score ###
        ##############################################################################
        scores_iter = {}
        poses_iter = {}
        for n, seq in enumerate(mpnn_out["generated_sequences"]):
            _pose_threaded = design_utils.thread_seq_to_pose(_pose2, seq)
            poses_iter[n] = design_utils.repack(_pose_threaded, sfx)
            scores_iter[n] = sfx(poses_iter[n])
            print(f"  Initial sequence {n} total_score: {scores_iter[n]}")

        best_score_id = min(scores_iter, key=scores_iter.get)
        #_pose = poses_iter[n].clone()
        _pose = poses_iter[best_score_id].clone()

        print(f"Relaxing initial guess sequence {best_score_id}")

        _pose2 = _pose.clone()
        fastRelax.apply(_pose2)

        print(f"Relaxed initial sequence: total_score = {_pose2.scores['total_score']}")

        ## Applying user-defined custom scoring
        scores_df = scoring.score_design(_pose2, pyrosetta.get_fa_scorefxn(), [])
        filt_scores = scoring.filter_scores(scores_df)

        results = {N_iter: {"pose": _pose2.clone(), "scores": scores_df.copy()}}

        print(f"Iter {N_iter} scores:\n{scores_df.iloc[0]}")
        N_iter += 1


    ####
    ## Done iterating, dumping outputs, if any
    ####
    if len(scoring.filter_scores(scores_df)) == 0:
        print(f"Design iteration {N} finished unsuccessfully in {(time.time() - iter_start_time):.2f} seconds.")
        continue

    print(f"Iter {N}, doing final proper relax and scoring")
    good_pose = _pose2.clone()

    _rlx_st = time.time()
    fastRelax_proper.apply(good_pose)
    print(f"Final relax finished after {(time.time()-_rlx_st):.2f} seconds.")

    ## Applying user-defined custom scoring
    scores_df = scoring.score_design(good_pose, pyrosetta.get_fa_scorefxn(), [])
    sfx(good_pose)
    scores_df.at[0, "description"] = output_name

    print(f"Design iteration {N} finished in {(time.time() - iter_start_time):.2f} seconds.")
    
    if len(scoring.filter_scores(scores_df)) != 0:
        print(f"Design iteration {N} is successful, dumping PDB: {output_name}.pdb")
        good_pose.dump_pdb(f"{output_name}.pdb")
        scoring_utils.dump_scorefile(scores_df, scorefilename)
