cd sxj680gallinahome
module load singularity
module load cuda/10.1
srun -p gpu --gpus 1 --mem 4000 --cpus-per-gpu 2 --pty bash
singularity run --nv --Bind {bind_path} {sif_path} train --input_dir {input_dir_path} --output_dir {output_dir_path}