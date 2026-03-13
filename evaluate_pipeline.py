import os
import argparse
import json
import traceback
from pathlib import Path

from evaluators.baseline import run_baseline_evaluation
from evaluators.pix2pix_cyclegan import run_model_evaluation
from evaluators.i2i_mamba import run_i2i_mamba_evaluation


def main():
    """Main orchestrator for PET model evaluation pipeline."""
    
    parser = argparse.ArgumentParser(

        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("data_dir", type=str, help="Directory containing .h5 files")
    parser.add_argument("--input-key", "-i", type=str, default="1_10")
    parser.add_argument("--target-key", "-t", type=str, default="full")
    parser.add_argument("--batch-size", "-b", type=int, default=1)
    parser.add_argument("--device", "-d", type=str, default=None, choices=["cuda", "cpu"])
    parser.add_argument("--checkpoints-dir", "-c", type=str, default=None)
    parser.add_argument("--save-dir", "-s", type=str, default="./evaluation_results")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["pix2pix", "cycle_gan", "i2i_mamba"],
        choices=["pix2pix", "cycle_gan", "i2i_mamba"],
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("PET RECONSTRUCTION EVALUATION PIPELINE")
    print("=" * 70 + "\n")

    all_results = {}

    # Step 1: Compute baseline comparison (lq vs hq)
    print("\n" + "=" * 70)
    print("STEP 1: BASELINE COMPARISON")
    print("=" * 70)

    try:
        baseline_results = run_baseline_evaluation(
            data_dir=args.data_dir,
            input_key=args.input_key,
            target_key=args.target_key,
            batch_size=args.batch_size,
            device=args.device,
            save_dir=args.save_dir,
            max_batches=args.max_batches,
        )
        all_results["baseline"] = baseline_results
        print("\n✓ Baseline computed\n")
    except Exception as e:
        print(f"Error computing baseline: {e}")
        traceback.print_exc()
        all_results["baseline"] = {"error": str(e)}

    # Step 2: Run evaluation for each model
    for model_name in args.models:
        print(f"\n[Model] {model_name}")
        try:
            if model_name == "i2i_mamba":
                results = run_i2i_mamba_evaluation(
                    data_dir=args.data_dir,
                    input_key=args.input_key,
                    target_key=args.target_key,
                    batch_size=args.batch_size,
                    device=args.device,
                    checkpoint_dir=args.checkpoints_dir,
                    save_dir=args.save_dir,
                    max_batches=args.max_batches,
                )
            else:
                # pix2pix and cycle_gan
                results = run_model_evaluation(
                    data_dir=args.data_dir,
                    model_name=model_name,
                    input_key=args.input_key,
                    target_key=args.target_key,
                    batch_size=args.batch_size,
                    device=args.device,
                    checkpoint_dir=args.checkpoints_dir,
                    save_dir=args.save_dir,
                    max_batches=args.max_batches,
                )
            all_results[model_name] = results
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            traceback.print_exc()
            all_results[model_name] = {"error": str(e)}

    # Step 3: Print summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for metric in ["psnr", "ssim", "lpips", "gmsd"]:
        print(f"\n{metric.upper()}:")
        base_key = f"{metric}_mean"

        baseline_result = all_results.get("baseline", {})
        has_baseline = base_key in baseline_result

        if has_baseline:
            base_mean = baseline_result[f"{metric}_mean"]
            base_std = baseline_result[f"{metric}_std"]
            print(f"  baseline   : {base_mean:.4f} ± {base_std:.4f}")

        for model_name in args.models:
            result = all_results.get(model_name, {})
            if "error" in result or base_key not in result:
                continue

            mean_val = result[f"{metric}_mean"]
            std_val = result[f"{metric}_std"]

            if has_baseline:
                delta = mean_val - base_mean
                if metric in ["lpips", "gmsd"]:
                    better = delta < 0
                else:
                    better = delta > 0
                tag = "better" if better else "worse"
                print(f"  {model_name:10s}: {mean_val:.4f} ± {std_val:.4f} ({delta:+.4f}, {tag})")
            else:
                print(f"  {model_name:10s}: {mean_val:.4f} ± {std_val:.4f}")

    # Step 4: Save summary to JSON
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        summary_path = os.path.join(args.save_dir, "comparison_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved: {summary_path}")

    if not args.checkpoints_dir:
        print("\nWarning: models were evaluated with random initialization unless the repo loaded checkpoints by default.")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
