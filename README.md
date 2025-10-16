# Modeling Offensive Language as a Distinct Class for Hate Speech Detection
This repository belongs to the Master's Thesis Project, "Modeling Offensive Language as a Distinct Class for Hate Speech Detection" by Areum Kim, supervised by Dr. Antske Fokkens and Dr. Hennie van der Vliet. The project explored how modeling offensive (but not hateful) language as a distinct class impacts the task of detection of hate speech. Using a ternary classification scheme (Hateful, Offensive, Clean), I evaluated a RoBERTa-base model in the full three-class setup and in binary variants where two classes are merged or the offensive class is removed (Hate vs. Non-hate, Non-clean vs. Clean, and Hate vs. Clean). 

# HateCheck-XR
In the project, to probe model behavior beyond set-internal performance, I revised both **HateCheck** [(Röttger et al., 2021)](https://aclanthology.org/2021.acl-long.4/) and an existing extension by [Khurana et al. (2025)](https://arxiv.org/abs/2410.15911), aligning them with the ternary system by re-annotating them and correcting errors present in the extension. The resulting dataset, **HateCheck-XR**, is available in this repository, under "dataset" in a csv format.

# Folder structure
```
Project
├─ hs_generalization/  
│  ├─ __init__.py
│  ├─ modes.py
│  ├─ train.py
│  ├─ test.py
│  └─ uitls.py    
├─ tools/
│  └─ run_many.py          
├─ configs/
│  └─ example.json
├─ dataset/
│ ├─ davidson/ # (expected location if using local copy)
│ └─ extended_hatecheck/reannotation.csv
├─ requirements.txt
└─ README.md
```


# Set-up
The code used in this study include modifications of Khurana et al. (2025)'s [code](https://github.com/urjakh/defverify), therefore it is advised to set up the environment as they did: 
```
# Create environment.
conda create -n hs-generalization python=3.9
conda activate hs-generalization

# Install packages.
python setup.py develop
pip install -r requirements.txt
```

# Training
Create a config file and run the following:
```
python -m hs_generalization.train -c configs\train\example.json
```

# Evaluation
Create a config file and run the following:

#running based on a single seed/checkpoint
```
python -m hs_generalization.test -c configs\test\example.json --dataset davidson --eval-mode 3class --train-mode 3class --seed 5 --checkpoint "outputs\davidson\RoBERTa-base\3class\RoBERTa-base_0.pt"

# If you want to run multiple files, for instance the best checkpoints from various seeds, you can use run_many.py:
python tools/run_many.py ^
  -c configs/test/test.json ^
  --dataset hatecheck_xr ^
  --eval-mode hate_nonhate ^
  --train-mode hate_clean ^
  --seeds 7 222 550 999 3111 ^
  --ckpt-pattern "outputs/davidson/RoBERTa-base/hate-clean/best/seed{seed}_*.pt" ^
  --hatecheck-csv dataset/extended_hatecheck/reannotation.csv
```




.
