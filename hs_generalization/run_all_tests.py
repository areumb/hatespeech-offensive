import subprocess
import os
import glob
import sys

# List of seeds as the folders are also created accordingly

#seeds=[7, 42, 111, 1234, 2025] #notclean clean
#seeds = [ 5, 11, 42, 100, 2021] #3classes
#seeds = [ 8, 94, 300, 1024, 2111] #hate-nonhate on hate-nonhate ------------SELECT each time!
seeds= [7, 222, 550, 999, 3111] #hate clean


#-----------------------------------------------------CHANGE it each time!
base_folder_pattern = "outputs/davidson/RoBERTa-base/hate-clean/best/seed{seed}_*.pt" #CHANGE each time!!!

# Path to test config file.
config_path = "configs/test/test.json"

for seed in seeds:
    # Build the folder path for the current seed.
    seed_folder = base_folder_pattern.format(seed=seed)
    
    # List all checkpoint files.
    checkpoint_files = glob.glob(seed_folder)
    print("checkpoint_files: ", checkpoint_files)
    

    # if the folder is empty, exit now
    if not checkpoint_files:
        print("No .pt files found --- aborting.")
        sys.exit(1)        # raises SystemExit and stops the whole script


    # Run tests for each checkpoint file found.
    for ckpt in checkpoint_files:

        print(f"Running test for seed {seed} using checkpoint: {ckpt}")

        subprocess.run([
            "python", "hs_generalization/test.py",
            "-c", config_path,
            "--seed", str(seed),
            "--checkpoint", ckpt
        ],
            check=True,
            stdin=subprocess.DEVNULL,   # <- no stdin, so no "press Enter" needed
        )
        print() 
        print(f"testing for seed {seed} using checkpoint {ckpt} done")
        print()        