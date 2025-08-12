import os
import argparse
from concurrent.futures import ThreadPoolExecutor
import subprocess
from typing import List

try:
    import torch
except Exception:
    torch = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiview shape generation across scenes in parallel over multiple GPUs.")
    parser.add_argument(
        "--input_folder",
        type=str,
        default="/home/simba/Documents/project/TRELLIS/datasets/ABO/renders_cond",
        help="Root folder containing scene subfolders with images.",
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="outputs_HOI3D_multi",
        help="Base output directory where per-scene outputs will be written.",
    )
    parser.add_argument(
        "--script",
        type=str,
        default="examples/shape_gen_multiview.py",
        help="Path to the generation script to execute per scene.",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default="7,14,21",
        help="Comma-separated image indices (0-based) to pick as front,left,back from the sorted image list.",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default="",
        help="Optional comma-separated list of scene folder names to process. If empty, all scenes in input_folder are used.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="",
        help="Comma-separated list of GPU IDs to use (as seen by nvidia-smi). If empty, will auto-detect all available GPUs.",
    )
    parser.add_argument(
        "--per_gpu_workers",
        type=int,
        default=1,
        help="Number of concurrent processes per GPU (1 recommended to avoid OOM).",
    )
    parser.add_argument(
        "--python",
        type=str,
        default="/home/simba/anaconda3/envs/hunyun/bin/python",
        help="Python binary to use when launching the generation script.",
    )
    return parser.parse_args()


def discover_gpus(user_gpus: str) -> List[int]:
    if user_gpus:
        return [int(x) for x in user_gpus.split(",") if x.strip() != ""]
    # Auto-detect via torch if available, else fall back to env or single GPU 0
    if torch is not None and torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return [int(x) for x in visible.split(",") if x.strip() != ""]
    # Default to GPU 0 if unsure
    return [0]


def list_scenes(input_folder: str, scenes_arg: str) -> List[str]:
    if scenes_arg:
        return [s for s in [x.strip() for x in scenes_arg.split(",")] if s]
    all_entries = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]
    all_entries.sort()
    return all_entries


def build_command(python_bin: str, script_path: str, input_folder: str, scene: str, indices: List[int], output_base: str) -> str:
    scene_dir = os.path.join(input_folder, scene)
    images = sorted(os.listdir(scene_dir))
    try:
        input1 = os.path.join(scene_dir, images[indices[0]])
        input2 = os.path.join(scene_dir, images[indices[1]])
        input3 = os.path.join(scene_dir, images[indices[2]])
    except IndexError:
        raise RuntimeError(f"Scene '{scene}' does not have enough images for indices {indices}.")
    sub_folder = "_".join([images[i].split(".")[0] for i in indices])
    output_dir = os.path.join(output_base, scene, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    return (
        f"{python_bin} {script_path} "
        f"--front_image '{input1}' --left_image '{input2}' --back_image '{input3}' "
        f"--output_dir '{output_dir}'"
    )


def run_command_on_gpu(cmd: str, gpu_id: int) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[GPU {gpu_id}] Launch: {cmd}")
    completed = subprocess.run(cmd, shell=True, env=env)
    return completed.returncode


def schedule_commands_across_gpus(commands: List[str], gpu_ids: List[int], per_gpu_workers: int = 1) -> None:
    # Distribute commands to GPUs in round-robin manner
    gpu_to_cmds = {g: [] for g in gpu_ids}
    for idx, cmd in enumerate(commands):
        gpu = gpu_ids[idx % len(gpu_ids)]
        gpu_to_cmds[gpu].append(cmd)

    # Launch per-GPU executors to enforce per-GPU concurrency cap
    executors = {g: ThreadPoolExecutor(max_workers=per_gpu_workers) for g in gpu_ids}
    futures = []
    try:
        for gpu, cmds in gpu_to_cmds.items():
            for cmd in cmds:
                futures.append(executors[gpu].submit(run_command_on_gpu, cmd, gpu))
        for fut in futures:
            rc = fut.result()
            if rc != 0:
                raise RuntimeError(f"A job failed with return code {rc}")
    finally:
        for ex in executors.values():
            ex.shutdown(wait=True)


if __name__ == "__main__":
    args = parse_args()

    indices = [int(x) for x in args.indices.split(",") if x.strip() != ""]
    if len(indices) != 3:
        raise ValueError("--indices must contain exactly three comma-separated integers (front,left,back)")

    gpu_ids = discover_gpus(args.gpus)
    print(f"Discovered GPUs: {gpu_ids}")

    scenes = list_scenes(args.input_folder, args.scenes)
    print(f"Total scenes to process: {len(scenes)}")

    commands: List[str] = []
    for scene in scenes:
        try:
            cmd = build_command(
                python_bin=args.python,
                script_path=args.script,
                input_folder=args.input_folder,
                scene=scene,
                indices=indices,
                output_base=args.output_base,
            )
            commands.append(cmd)
        except RuntimeError as e:
            print(f"Skipping scene '{scene}': {e}")

    if not commands:
        print("No commands to run. Exiting.")
    else:
        schedule_commands_across_gpus(commands, gpu_ids, per_gpu_workers=args.per_gpu_workers)
        print("All jobs finished successfully.")
