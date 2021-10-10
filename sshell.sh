srun --gpus-per-task=2 --time=0 --cpus-per-task=1 --mem-per-cpu=20G  --qos=abc_normal --partition=abc --account=co_abc  --nodes=1 --pty /bin/bash
module load cuda/10.0
module load cudnn/7.5
