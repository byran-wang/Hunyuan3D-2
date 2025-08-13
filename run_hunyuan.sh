eval "$(conda shell.bash hook)"
conda activate trellis
DATA_DIR="/home/simba/Documents/project/TRELLIS/datasets/ABO"   
OUTPUT_DIR="outputs_HOI3D_multi"

python run_hunyuan.py --seq_list all --execute_list inference --process_list evaluation --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} --rebuild --max_workers 100
python run_hunyuan.py --seq_list all --execute_list inference --process_list eval_summary --data_dir ${DATA_DIR} --output_dir ${OUTPUT_DIR} --rebuild