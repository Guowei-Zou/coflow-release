#!/usr/bin/env python
"""
Evaluation script for MAMuJoCo with corrected test_ret values.
Results will be saved to: results_corrected_testret/

Usage:
    # Evaluate all available checkpoints
    python eval_mamujoco_corrected.py -g 0 --num_eval 10

    # Evaluate specific checkpoint
    python eval_mamujoco_corrected.py -g 0 --load_steps 500000 --num_eval 10

    # Evaluate multiple checkpoints
    python eval_mamujoco_corrected.py -g 0 --load_steps 50000,100000,500000 --num_eval 10

    # Evaluate specific environment
    python eval_mamujoco_corrected.py -g 0 --env 2ant --quality Good --ctde False
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import diffuser.utils as utils
from diffuser.utils.launcher_util import build_config_from_dict


# Corrected test_ret values based on normalized expert returns
CORRECTED_TEST_RET = {
    # 2ant: expert_return / returns_scale
    "2ant-Good": 6.7,      # 2556.03 / 380 = 6.73
    "2ant-Medium": 3.3,    # 1061.27 / 320 = 3.32
    "2ant-Poor": 2.6,      # 393.64 / 150 = 2.62
    # 2halfcheetah
    "2halfcheetah-Good": 6.9,    # 6924.11 / 1000 = 6.92
    "2halfcheetah-Medium": 5.0,  # 1485.00 / 300 = 4.95
    "2halfcheetah-Poor": 4.0,    # 400.45 / 100 = 4.00
    # 4ant
    "4ant-Good": 7.3,      # 2754.34 / 380 = 7.25
    "4ant-Medium": 4.6,    # 1457.71 / 320 = 4.56
    "4ant-Poor": 2.8,      # 415.99 / 150 = 2.77
}

# Model directory patterns
MODEL_PATTERNS = {
    ("2ant", "Good", False): "h_10-hh_18-models.SharedConvAttentionDeconv-r_380.0-dl_datasets.SequenceDataset-ctde_False-meanflow",
    ("2ant", "Good", True): "h_10-hh_18-models.SharedConvAttentionDeconv-r_380.0-dl_datasets.SequenceDataset-ctde_True-meanflow",
    ("2ant", "Medium", False): "h_10-hh_18-models.SharedConvAttentionDeconv-r_320.0-dl_datasets.SequenceDataset-ctde_False-meanflow",
    ("2ant", "Medium", True): "h_10-hh_18-models.SharedConvAttentionDeconv-r_320.0-dl_datasets.SequenceDataset-ctde_True-meanflow",
    ("2ant", "Poor", False): "h_10-hh_18-models.SharedConvAttentionDeconv-r_150.0-dl_datasets.SequenceDataset-ctde_False-meanflow",
    ("2ant", "Poor", True): "h_10-hh_18-models.SharedConvAttentionDeconv-r_150.0-dl_datasets.SequenceDataset-ctde_True-meanflow",
    ("2halfcheetah", "Good", False): "h_10-hh_18-models.SharedConvAttentionDeconv-r_1000.0-ctde_False-meanflow",
    ("2halfcheetah", "Good", True): "h_10-hh_18-models.SharedConvAttentionDeconv-r_1000.0-ctde_True-meanflow",
    ("2halfcheetah", "Medium", False): "h_10-hh_18-models.SharedConvAttentionDeconv-r_300.0-ctde_False-meanflow",
    ("2halfcheetah", "Medium", True): "h_10-hh_18-models.SharedConvAttentionDeconv-r_300.0-ctde_True-meanflow",
    ("2halfcheetah", "Poor", False): "h_10-hh_18-models.SharedConvAttentionDeconv-r_100.0-ctde_False-meanflow",
    ("2halfcheetah", "Poor", True): "h_10-hh_18-models.SharedConvAttentionDeconv-r_100.0-ctde_True-meanflow",
    ("4ant", "Good", False): "h_10-hh_18-models.SharedConvAttentionDeconv-r_380.0-ctde_False-meanflow",
    ("4ant", "Good", True): "h_10-hh_18-models.SharedConvAttentionDeconv-r_380.0-ctde_True-meanflow",
    ("4ant", "Medium", False): "h_10-hh_18-models.SharedConvAttentionDeconv-r_320.0-ctde_False-meanflow",
    ("4ant", "Medium", True): "h_10-hh_18-models.SharedConvAttentionDeconv-r_320.0-ctde_True-meanflow",
    ("4ant", "Poor", False): "h_10-hh_18-models.SharedConvAttentionDeconv-r_150.0-ctde_False-meanflow",
    ("4ant", "Poor", True): "h_10-hh_18-models.SharedConvAttentionDeconv-r_150.0-ctde_True-meanflow",
}


def find_seed_dirs(model_path):
    """Return trained seed run directories under a model directory."""
    seed_dirs = []
    for name in sorted(os.listdir(model_path)):
        seed_path = os.path.join(model_path, name)
        if not os.path.isdir(seed_path) or not name.isdigit():
            continue
        if os.path.exists(os.path.join(seed_path, "parameters.pkl")):
            seed_dirs.append(seed_path)
    return seed_dirs


def run_dirs_for_model(model_dir):
    seed_dirs = find_seed_dirs(model_dir)
    if seed_dirs:
        return seed_dirs
    if os.path.exists(os.path.join(model_dir, "parameters.pkl")):
        return [model_dir]
    return []


def find_model_dirs(base_dir, env, quality, ctde):
    """Find trained run directories for a given configuration."""
    dataset_dir = os.path.join(base_dir, f"{env}-{quality}")
    if not os.path.exists(dataset_dir):
        return []

    # Try to find matching pattern
    key = (env, quality, ctde)
    if key in MODEL_PATTERNS:
        model_subdir = MODEL_PATTERNS[key]
        model_dir = os.path.join(dataset_dir, model_subdir)
        if os.path.exists(model_dir):
            return run_dirs_for_model(model_dir)

    # Fallback: search by pattern
    ctde_str = "ctde_True" if ctde else "ctde_False"
    run_dirs = []
    for subdir in os.listdir(dataset_dir):
        if ctde_str in subdir:
            model_dir = os.path.join(dataset_dir, subdir)
            run_dirs.extend(run_dirs_for_model(model_dir))

    return run_dirs


def find_available_checkpoints(log_dir):
    """Find all available checkpoint steps in the log directory."""
    ckpt_dir = os.path.join(log_dir, "checkpoint")
    if not os.path.exists(ckpt_dir):
        return []

    steps = []
    for f in os.listdir(ckpt_dir):
        if f.startswith("state_") and f.endswith(".pt"):
            try:
                step = int(f.replace("state_", "").replace(".pt", ""))
                steps.append(step)
            except ValueError:
                continue

    return sorted(steps)


def evaluate_single(log_dir, load_step, num_eval, test_ret, condition_guidance_w=None,
                    results_subdir="results_corrected_testret", overwrite=False):
    """Run evaluation for a single model with corrected test_ret."""

    # Check checkpoint
    ckpt_file = os.path.join(log_dir, f"checkpoint/state_{load_step}.pt")
    if not os.path.exists(ckpt_file):
        print(f"[SKIP] Checkpoint not found: {ckpt_file}")
        return None

    # Create results directory
    results_dir = os.path.join(log_dir, results_subdir)
    os.makedirs(results_dir, exist_ok=True)

    # Check if results exist
    result_file = os.path.join(
        results_dir,
        f"step_{load_step}-ep_{num_eval}-testret_{test_ret}.json"
    )
    if condition_guidance_w is not None:
        result_file = result_file.replace(".json", f"-cg_{condition_guidance_w}.json")

    if not overwrite and os.path.exists(result_file):
        print(f"[SKIP] Results already exist: {result_file}")
        return None

    # Load original config
    params_file = os.path.join(log_dir, "parameters.pkl")
    if not os.path.exists(params_file):
        print(f"[SKIP] Parameters file not found: {params_file}")
        return None

    with open(params_file, "rb") as f:
        params = pickle.load(f)

    original_test_ret = params["Config"].get("test_ret", 1.0)
    print(f"\n{'='*60}")
    print(f"Log dir: {log_dir}")
    print(f"Original test_ret: {original_test_ret}")
    print(f"Corrected test_ret: {test_ret}")
    print(f"{'='*60}\n")

    # Initialize evaluator
    evaluator_config = utils.Config("utils.MADEvaluator", verbose=True)
    evaluator = evaluator_config()

    # Pass corrected test_ret to evaluator.init()
    evaluator.init(
        log_dir=log_dir,
        num_eval=num_eval,
        num_envs=num_eval,
        condition_guidance_w=condition_guidance_w,
        use_ddim_sample=False,
        n_ddim_steps=5,
        test_ret=test_ret,  # Pass corrected test_ret here!
    )

    # Run evaluation
    metrics = evaluator.evaluate(load_step=load_step)

    # Save results to new directory
    if metrics is not None:
        result_data = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in metrics.items()
        }
        result_data["test_ret"] = test_ret
        result_data["original_test_ret"] = original_test_ret

        with open(result_file, "w") as f:
            json.dump(result_data, f, indent=2)
        print(f"[DONE] Results saved to: {result_file}")

    # Cleanup evaluator
    del evaluator

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate MAMuJoCo models with corrected test_ret")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--load_steps", type=str, default=None,
                        help="Checkpoint steps to load (comma-separated, e.g., '50000,100000,500000'). "
                             "If not specified, evaluates all available checkpoints.")
    parser.add_argument("--num_eval", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--env", type=str, default=None, help="Specific environment (2ant, 2halfcheetah, 4ant)")
    parser.add_argument("--quality", type=str, default=None, help="Data quality (Good, Medium, Poor)")
    parser.add_argument("--ctde", type=str, default=None, help="CTDE mode (True/False)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    parser.add_argument("--cg", type=float, default=None, help="Condition guidance weight")
    parser.add_argument(
        "--base_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs",
            "ma_meanflow_mamujoco",
        ),
        help="Root directory containing trained MA-MuJoCo logs.",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    base_dir = args.base_dir

    # Determine which models to evaluate
    if args.env is not None:
        envs = [args.env]
    else:
        envs = ["2ant", "2halfcheetah", "4ant"]

    if args.quality is not None:
        qualities = [args.quality]
    else:
        qualities = ["Good", "Medium", "Poor"]

    if args.ctde is not None:
        ctde_modes = [args.ctde.lower() == "true"]
    else:
        ctde_modes = [False, True]

    # Parse load_steps if specified
    specified_steps = None
    if args.load_steps is not None:
        specified_steps = [int(s.strip()) for s in args.load_steps.split(",")]

    # Run evaluations
    results_summary = []
    for env in envs:
        for quality in qualities:
            for ctde in ctde_modes:
                dataset_name = f"{env}-{quality}"
                test_ret = CORRECTED_TEST_RET.get(dataset_name)
                if test_ret is None:
                    print(f"[SKIP] No corrected test_ret for {dataset_name}")
                    continue

                log_dirs = find_model_dirs(base_dir, env, quality, ctde)
                if not log_dirs:
                    print(f"[SKIP] Model directories not found for {dataset_name} (ctde={ctde})")
                    continue

                mode = "ctde" if ctde else "attn"
                for log_dir in log_dirs:
                    # Get checkpoint steps to evaluate
                    if specified_steps is not None:
                        load_steps = specified_steps
                    else:
                        load_steps = find_available_checkpoints(log_dir)
                        if not load_steps:
                            print(f"[SKIP] No checkpoints found for {dataset_name} (ctde={ctde})")
                            continue

                    print(f"\n{'#'*70}")
                    print(f"# Evaluating: {dataset_name} ({mode})")
                    print(f"# Run dir: {log_dir}")
                    print(f"# Checkpoints: {load_steps}")
                    print(f"{'#'*70}")

                    for load_step in load_steps:
                        try:
                            metrics = evaluate_single(
                                log_dir=log_dir,
                                load_step=load_step,
                                num_eval=args.num_eval,
                                test_ret=test_ret,
                                condition_guidance_w=args.cg,
                                overwrite=args.overwrite,
                            )
                            if metrics is not None:
                                results_summary.append({
                                    "env": env,
                                    "quality": quality,
                                    "mode": mode,
                                    "step": load_step,
                                    "test_ret": test_ret,
                                    "metrics": metrics,
                                })
                        except Exception as e:
                            print(f"[ERROR] Evaluation failed for step {load_step}: {e}")
                            import traceback
                            traceback.print_exc()

    # Print summary
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    for r in results_summary:
        avg_reward = r["metrics"].get("average_ep_reward", "N/A")
        if isinstance(avg_reward, (list, np.ndarray)):
            avg_reward = np.mean(avg_reward)
        print(f"{r['env']}-{r['quality']} ({r['mode']}) step={r['step']}: test_ret={r['test_ret']}, avg_reward={avg_reward:.2f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
