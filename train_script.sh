conda activate cs234

# Run the training script and shut down the VM after it completes
nohup bash -c 'python train.py > output.txt 2>&1; sudo shutdown -h now' &