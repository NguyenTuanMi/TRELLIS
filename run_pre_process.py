
import argparse
import json
from dataclasses import asdict

from restructure_prompt import rewrite_prompt


def main():
    parser = argparse.ArgumentParser(description="Run the full scale->mass->sim-export chain on an existing GLB.")
    parser.add_argument("--user_prompt", type=str, default=None, required=True),
    parser.add_argument("--rewrite_prompt", action="store_true")
    parser.add_argument("--export_sim_ready", action="store_true")
    parser.add_argument("--vlm_base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.user_prompt is not None:
        print(f"\n[1/2] Rewriting Prompt ...")
        prompt_result = rewrite_prompt(
            user_prompt=args.user_prompt,
            vlm_base_url=args.vlm_base_url,
            vlm_model=args.vlm_model,
        )
        print(json.dumps(asdict(prompt_result), indent=2))
    else:
        print("\n[1/2] Error: No input prompt")


if __name__ == "__main__":
    main()