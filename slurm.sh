#!/bin/bash

# Make sure this file has Unix line endings (LF) and is run with bash:
#   chmod +x slurm.sh
#   ./slurm.sh
#
# This script attempts to connect to a GPU cluster using srun.
# It will automatically retry if it connects to 'dlcgpu07' or 'dlcgpu04',
# and will keep the connection open once it lands on a different node.

echo "Starting GPU cluster connection attempts..."

while true; do
    echo "--------------------------------------------------------"
    echo "Attempting to connect to a GPU node for initial check..."

    # Step 1: Allocate a node and check its hostname without --pty.
    # This ensures the exit status of the inner command is correctly propagated.
    # We use --job-name for easier identification in squeue.
    # The 'hostname' command's output will be captured to determine the allocated node.
    srun_check_output=$(srun --partition=dllabdlc_gpu-rtx2080 --gres=gpu:1 --job-name=gpu_check hostname 2>&1)
    SRUN_CHECK_EXIT_STATUS=$? # Capture the exit status of the srun command

    # Extract the hostname from the srun_check_output.
    # 'tail -n 1' gets the last line (usually the hostname), and 'tr -d '\n\r'' removes newlines.
    NODE_HOSTNAME=$(echo "$srun_check_output" | tail -n 1 | tr -d '\n\r')

    echo "Initial check connected to node: $NODE_HOSTNAME"

    # Check if the connected node is a restricted one (dlcgpu07 or dlcgpu04).
    if [[ "$NODE_HOSTNAME" = "dlcgpu07" || "$NODE_HOSTNAME" = "dlcgpu04" ]]; then
        echo "Detected restricted GPU ($NODE_HOSTNAME). Retrying in 5 seconds..."
        sleep 5 # Wait before retrying the srun command
    elif [ -z "$NODE_HOSTNAME" ] || [ $SRUN_CHECK_EXIT_STATUS -ne 0 ]; then
        # Handle cases where hostname might be empty or srun failed to allocate.
        echo "Failed to get a hostname or srun allocation failed. Retrying in 5 seconds..."
        echo "srun output: $srun_check_output" # Print srun output for debugging
        sleep 5
    else
        echo "--------------------------------------------------------"
        echo "Successfully found a desired GPU node: $NODE_HOSTNAME."
        echo "Now launching interactive session on $NODE_HOSTNAME..."

        # Step 2: Launch an interactive session on the found desired node.
        # We use --nodelist to explicitly request the previously identified node.
        # This ensures we get the same node for the interactive session.
        # The script will hand over control to this interactive session.
        srun --partition=dllabdlc_gpu-rtx2080 --gres=gpu:1 --pty --nodelist="$NODE_HOSTNAME" bash

        # The script will effectively stop here and hand over control to the interactive session.
        # If the interactive session exits, the outer script will also exit.
        break # Exit the while loop as the interactive session is successfully launched
    fi
done

echo "Script finished. You should be in your GPU session."