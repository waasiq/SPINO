List all GPUS: sinfo -p dllabdlc_gpu-rtx2080 -o "%n" 
See which GPU's are free: sinfo -p dllabdlc_gpu-rtx2080 -N -o "%10N %10T %20G %E"

Connect to GPU using srun: srun --partition=dllabdlc_gpu-rtx2080 --gres=gpu:1 --pty bash
Can also ssh to already connected gpu


# To don't allow idling of the system, do this:

while true; do
    echo "Keep-alive: $(date)"
    nvidia-smi > /dev/null 2>&1
    sleep 300
done &
[3] 3536125