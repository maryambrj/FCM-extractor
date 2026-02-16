#!/usr/bin/env python3
"""
Generic FCM evaluation script.

Compares predicted FCMs against ground-truth FCMs using the soft_measures
scoring library. Handles directory structure differences by temporarily
flattening nested FCM output folders.

Usage:
    python run_evaluation.py \
        --gt-dir results/.../ground_truth \
        --pred-dir results/.../fcm_outputs/gpt-5

The script will:
1. Create a temporary flat directory with copies of predicted FCMs.
2. Run the soft_measures scoring (loads model once, scores all pairs).
3. Save results next to the ground_truth / pred directories.
4. Clean up the temporary directory.
"""

import os
import sys
import shutil
import glob
import argparse
import tempfile
from pathlib import Path

# Add soft_measures to path so we can import from it
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "soft_measures"))

from compare_fcm_directories import compare_directories


def flatten_predictions(pred_dir: str, temp_dir: str) -> dict:
    """
    Copy predicted FCMs from a nested structure into a flat directory,
    renaming them to match ground-truth file names.

    Expected input structure:
        pred_dir/
            BD006/
                BD006_fcm.json
            BD007/
                BD007_fcm.json

    Output structure:
        temp_dir/
            BD006.json
            BD007.json

    Returns:
        dict: Mapping of {dataset_id: original_path} for provenance tracking.
    """
    provenance = {}

    # Pattern 1: Nested folders with *_fcm.json files
    for fcm_file in glob.glob(os.path.join(pred_dir, "*", "*_fcm.json")):
        basename = os.path.basename(fcm_file)          # e.g. BD006_fcm.json
        dataset_id = basename.replace("_fcm.json", "")  # e.g. BD006
        dest = os.path.join(temp_dir, f"{dataset_id}.json")
        shutil.copy2(fcm_file, dest)
        provenance[dataset_id] = fcm_file

    # Pattern 2: Flat JSON/CSV files already at the top level (no renaming needed)
    if not provenance:
        for ext in ("*.json", "*.csv"):
            for f in glob.glob(os.path.join(pred_dir, ext)):
                basename = os.path.basename(f)
                shutil.copy2(f, os.path.join(temp_dir, basename))
                stem = Path(f).stem
                provenance[stem] = f

    return provenance


def infer_output_dir(gt_dir: str, pred_dir: str) -> str:
    """
    Infer a sensible output directory from the input paths.

    If gt_dir and pred_dir share a common parent (e.g. .../high-quality-data/),
    put the results there. Otherwise, fall back to the pred_dir's parent.
    """
    gt_parent = os.path.dirname(os.path.abspath(gt_dir))
    pred_parent = os.path.dirname(os.path.abspath(pred_dir))

    # Check for a shared ancestor (e.g. both under .../high-quality-data/)
    if gt_parent == pred_parent:
        base = gt_parent
    else:
        # Go up from pred_dir: .../fcm_outputs/gpt-5 -> .../fcm_outputs -> ...
        base = os.path.dirname(pred_parent)

    model_name = os.path.basename(os.path.abspath(pred_dir))
    return os.path.join(base, "evaluation_results", model_name)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predicted FCMs against ground-truth using soft_measures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluation.py \\
      --gt-dir results/biodiversity-data/without-clustering/high-quality-data/ground_truth \\
      --pred-dir results/biodiversity-data/without-clustering/high-quality-data/fcm_outputs/gpt-5

  python run_evaluation.py \\
      --gt-dir path/to/ground_truth \\
      --pred-dir path/to/predictions \\
      --output-dir path/to/save/results
        """
    )
    parser.add_argument("--gt-dir", required=True,
                        help="Directory containing ground-truth FCMs (CSV or JSON)")
    parser.add_argument("--pred-dir", required=True,
                        help="Directory containing predicted FCMs (nested or flat)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save results (auto-inferred if not provided)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Similarity threshold")
    parser.add_argument("--model-name", default="Qwen/Qwen3-Embedding-0.6B",
                        help="Embedding model for scoring (default: Qwen/Qwen3-Embedding-0.6B)")
    parser.add_argument("--tp-scale", type=float, default=1.0,
                        help="True positive scale (default: 1.0)")
    parser.add_argument("--pp-scale", type=float, default=0.6,
                        help="Partial positive scale")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output-format", choices=["csv", "json", "both"],
                        default="both", help="Output format (default: both)")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isdir(args.gt_dir):
        print(f"❌ Ground-truth directory not found: {args.gt_dir}")
        sys.exit(1)
    if not os.path.isdir(args.pred_dir):
        print(f"❌ Predictions directory not found: {args.pred_dir}")
        sys.exit(1)

    # Determine output directory
    output_dir = args.output_dir or infer_output_dir(args.gt_dir, args.pred_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("FCM EVALUATION")
    print("=" * 60)
    print(f"  Ground Truth:  {os.path.abspath(args.gt_dir)}")
    print(f"  Predictions:   {os.path.abspath(args.pred_dir)}")
    print(f"  Output:        {os.path.abspath(output_dir)}")
    print(f"  Threshold:     {args.threshold}")
    print(f"  Model:         {args.model_name}")
    print("=" * 60)
    print()

    # Step 1: Create temporary flat directory
    temp_dir = tempfile.mkdtemp(prefix="fcm_eval_flat_")
    print(f"📁 Created temporary directory: {temp_dir}")

    try:
        # Step 2: Flatten predictions into temp directory
        print("📋 Copying and flattening predictions...")
        provenance = flatten_predictions(args.pred_dir, temp_dir)

        if not provenance:
            print("❌ No FCM files found in predictions directory!")
            sys.exit(1)

        print(f"   Found {len(provenance)} FCM files:")
        for dataset_id, original_path in sorted(provenance.items()):
            print(f"     {dataset_id} ← {os.path.relpath(original_path)}")
        print()

        # Step 3: Run scoring
        print("🔬 Running evaluation...")
        print()
        results = compare_directories(
            dir1=args.gt_dir,
            dir2=temp_dir,
            output_dir=output_dir,
            output_format=args.output_format,
            threshold=args.threshold,
            model_name=args.model_name,
            tp_scale=args.tp_scale,
            pp_scale=args.pp_scale,
            batch_size=args.batch_size,
            seed=args.seed,
            verbose=True
        )

        # Step 4: Print summary
        if not results.empty:
            print()
            print("=" * 60)
            print("📊 EVALUATION SUMMARY")
            print("=" * 60)
            print(f"  Pairs evaluated: {len(results)}")
            print(f"  Mean F1:         {results['F1'].mean():.4f}")
            print(f"  Mean Jaccard:    {results['Jaccard'].mean():.4f}")
            print(f"  Min F1:          {results['F1'].min():.4f}")
            print(f"  Max F1:          {results['F1'].max():.4f}")
            print(f"  Results saved:   {os.path.abspath(output_dir)}")
            print("=" * 60)
        else:
            print("⚠️  No pairs were evaluated. Check that file names match.")

    finally:
        # Step 5: Clean up temporary directory
        print(f"\n🧹 Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("✅ Done!")


if __name__ == "__main__":
    main()
