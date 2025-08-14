import argparse
import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def get_metadata(data_dir):
    metadata_file = os.path.join(data_dir, "metadata.csv")
    metadata = pd.read_csv(metadata_file)
    return metadata

class run_trellis:
    def __init__(self, args, extras):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.seq_list = args.seq_list
        self.execute_list = args.execute_list
        self.process_list = args.process_list
        self.data_dir = args.data_dir
        self.output_dir = args.output_dir
        self.rebuild = args.rebuild
        self.vis = args.vis
        self.extras = extras
        self.trellis_python = "/home/simba/anaconda3/envs/trellis/bin/python"
        # Get max_workers from extras or use default
        self.max_workers = extras.get('max_workers', 8)
        self.process_mapping = {
            "inference": {
                "single_image_to_3D": self.single_image_to_3D,
                "evaluation": self.evaluation,
                "eval_summary": self.eval_summary,
            },
        }

    def run(self):
        for exe in self.execute_list:
            for process in self.process_list:
                if process == "eval_summary" or process == "evaluation":
                    self.process_mapping[exe][process](self.seq_list[0], **self.extras)
                else:
                    for seq in self.seq_list:
                        self.process_mapping[exe][process](seq, **self.extras)

    def evaluation(self, scene_name, **kwargs):
        self.print_header("evaluation")
        metadata = get_metadata(self.data_dir)
        scenes = os.listdir(self.output_dir)
        
        # Collect all evaluation tasks
        evaluation_tasks = []
        for scene in scenes:
            if os.path.exists(f"{self.output_dir}/{scene}/eval") and not self.rebuild:
                print(f"[warning] Scene {scene} already evaluated")
                continue
                
            views = os.listdir(os.path.join(self.output_dir, scene))
            for view in views:
                if scene not in metadata['sha256'].values:
                    print(f"Scene {scene} not found in metadata")
                    continue
                    
                scene_metadata = metadata[metadata['sha256'] == scene]
                object_gt = os.path.join(self.data_dir, "renders", scene_metadata["sha256"].values[0], "mesh.ply")
                if not os.path.exists(object_gt):
                    print(f"[warning] GT object {object_gt} not found in metadata")
                    continue
                    
                object_pred = os.path.join(self.output_dir, scene, view, "mesh.obj")
                if not os.path.exists(object_pred):
                    print(f"[warning] Pred object {object_pred} not found in metadata")
                    continue
                    
                evaluation_tasks.append((scene, view, object_gt, object_pred))
        
        print(f"Total evaluation tasks: {len(evaluation_tasks)}")
        print(f"Using {self.max_workers} parallel workers")
        
        # Process tasks in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._evaluate_single, scene, view, object_gt, object_pred): (scene, view)
                for scene, view, object_gt, object_pred in evaluation_tasks
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(evaluation_tasks), desc="Evaluating scenes/views") as pbar:
                for future in as_completed(future_to_task):
                    scene, view = future_to_task[future]
                    try:
                        result = future.result()
                        pbar.set_postfix_str(f"{scene}/{view}")
                        pbar.update(1)
                    except Exception as exc:
                        print(f"Task {scene}/{view} generated an exception: {exc}")
                        pbar.update(1)

    def _evaluate_single(self, scene, view, object_gt, object_pred):
        """Evaluate a single scene/view combination"""
        if self.rebuild:
            os.system(f"rm -rf {self.output_dir}/{scene}/{view}/eval")
            
        cmd = ""
        cmd += f"cd {self.current_dir} && "
        cmd += f"{self.trellis_python} evaluate.py "
        cmd += f"--object_gt {object_gt} "
        cmd += f"--object_pred {object_pred} "
        cmd += f"--seq_name {scene}_{view} "
        cmd += f"--out_dir {self.output_dir}/{scene}/{view}/eval "
        print(cmd)
        
        # Use subprocess instead of os.system for better control
        import subprocess
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error evaluating {scene}/{view}: {result.stderr}")
            return result.returncode == 0
        except Exception as e:
            print(f"Exception evaluating {scene}/{view}: {e}")
            return False

    def single_image_to_3D(self, scene_name, **kwargs):
        self.print_header("single_image_to_3D")
        pass

    def eval_summary(self, scene_name, **kwargs):
        self.print_header("eval_summary")
        if args.rebuild:
            os.system(f"rm -rf {self.output_dir}/metrics_summary/")

        cmd = ""
        cmd += f"cd {self.current_dir} && "
        cmd += f"{self.trellis_python} extract_jsons.py --parent_dir {self.output_dir} --metric_folder eval --out_dir {self.output_dir}/metrics_summary/ "
        print(cmd)
        os.system(cmd)

    def print_header(self, process):
        header = f"========== start: {process} =========="
        print("-"*len(header))
        print(header)
        print("-"*len(header))    

def main(args, extras):
    # Convert extras list to dictionary
    extras_dict = {}
    for i in range(0, len(extras), 2):
        if i + 1 < len(extras):
            key = extras[i].lstrip('-')  # Remove leading dashes
            value = extras[i + 1]
            # Try to convert value to int or float if possible
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass
            extras_dict[key] = value
    
    # Add max_workers from command line args
    extras_dict['max_workers'] = args.max_workers
    
    run_trellis(args, extras_dict).run()



if __name__ == "__main__":
    all_sequences = [
        "4f4616dc1b2be598e4332130aee5be906b4ea7ce81d8f6b8cda1e59bd1ae1526",
        "7ab52011f82f60e5d06b5582cd874bf6a7b0288489fe3827aec0829dfee71399",
        "03554445ec87ee6dc62ae789015c51fa7a4f4f6802f3b87bc9ce090775620854",
        "bc8d0eb8f1525ee9ab39b00a197980f14ffce1b7d83531373e62696f28c18852",
        "f5548e475296967ad9797bc51fa23738648c2def23082dce6fd300c271439b69",   
    ]

    parser = argparse.ArgumentParser()

    parser.add_argument('--seq_list',
        choices=all_sequences + ['all'],
        help="Specify the sequence list. Use 'all' to select all sequences.",
        nargs='+',  # To accept multiple values in a list
        required=True  # This makes the argument mandatory
    )

    parser.add_argument('--execute_list', 
        choices=[
                "data_preprocess",
                "finetune",
                "inference"
                ], 
        help="Specify the execution option.", 
        nargs='+',  # To accept multiple values in a list
        required=False  # This makes the argument mandatory
    )
    parser.add_argument('--process_list', 
        choices=["single_image_to_3D", 
                "evaluation",
                "eval_summary"
                ],
        help="Specify the process option.", 
        nargs='+',  # To accept multiple values in a list
        required=True  # This makes the argument mandatory
    )
    parser.add_argument('--rebuild', action='store_true', help='Rebuild the process')
    parser.add_argument('--vis', action='store_true', help='Visualize the process')
    parser.add_argument('--data_dir', type=str, default="datasets/ABO", help='Dataset directory')
    parser.add_argument('--output_dir', type=str, default="outputs/outputs_HOI3D", help='Output directory')
    parser.add_argument('--max_workers', type=int, default=8, help='Maximum number of parallel workers for evaluation (default: 4)')


  
    args, extras = parser.parse_known_args()
    if 'all' in args.seq_list:
        args.seq_list = all_sequences

    main(args, extras)