List all GPUS: sinfo -p dllabdlc_gpu-rtx2080 -o "%n" 
See which GPU's are free: sinfo -p dllabdlc_gpu-rtx2080 -N -o "%10N %10T %20G %E"
