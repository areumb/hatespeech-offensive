# Modeling Offensive Language as a Distinct Class for Hate Speech Detection
This repository is part of my master's thesis project,"Modeling Offensive Language as a Distinct Class for Hate Speech Detection" [(Kim, 2025)](./Thesis_Areum.pdf), supervised by Dr. Antske Fokkens and Dr. Hennie van der Vliet. The project explored how modeling offensive (but not hateful) language as a distinct class impacts the task of detection of hate speech. Using a ternary classification scheme (Hateful, Offensive, Clean), I fine-tuned and evaluated a RoBERTa-base model in the full three-class setup and in binary variants where two classes are merged or the offensive class is removed (Hate vs. Non-hate, Non-clean vs. Clean, and Hate vs. Clean). The code used in this study includes my modifications and extensions of [Khurana et al. (2025)](https://arxiv.org/abs/2410.15911)'s [code](https://github.com/urjakh/defverify).

# HateCheck-XR
In the project, to probe model behavior beyond set-internal performance, I revised both **HateCheck** [(Röttger et al., 2021)](https://aclanthology.org/2021.acl-long.4/) and an existing extension by [Khurana et al. (2025)](https://aclanthology.org/2025.coling-main.293/), aligning them with the ternary system by re-annotating them and correcting errors present in the extension. The resulting dataset, **HateCheck-XR** (3,855 cases, 37 functionalities), is available in this repository at [`datasets/hatecheck-xr/hatecheck-xr.csv`](./datasets/hatecheck-xr/hatecheck-xr.csv) (semicolon-separated).

# Folder structure
```
Project
├─ configs/
│  ├─ example.json           # quick smoke-test config
│  ├─ train/example.json     # training config
│  └─ test/example.json      # evaluation config with {seed} placeholders
├─ datasets/
│  ├─ davidson/              # place the prepared Davidson HF dataset here (not distributed)
│  └─ hatecheck-xr/hatecheck-xr.csv
├─ hs_generalization/
│  ├─ __init__.py
│  ├─ modes.py           # class schemes (3class / hate_nonhate / nonclean_clean / hate_clean) & logit projections
│  ├─ train.py           # fine-tuning pipeline
│  ├─ test.py            # unified evaluator (Davidson test split & HateCheck-XR)
│  ├─ run_many.py        # runs test.py across seeds/checkpoints
│  └─ utils.py
├─ scripts/
│  └─ create_hf_dataset.py   # builds the HF dataset 
├─ outputs/                  # checkpoints of the fine-tuned model (create on your own)
├─ README.md
├─ Thesis_Areum.pdf          
├─ requirements.txt
└─ setup.py
```

Label encoding used throughout: `0 = hateful, 1 = offensive, 2 = clean`.

# Set-up
Set up the environment like the following:
```
# Create environment.
conda create -n hs-generalization python=3.9
conda activate hs-generalization

# Install packages.
pip install -e .
pip install -r requirements.txt
```

# Data preparation
The Davidson dataset is not redistributed here. Download `labeled_data.csv` from the [official repository](https://github.com/t-davidson/hate-speech-and-offensive-language) and build the HuggingFace dataset. For example:
```
python scripts/create_hf_dataset.py -n davidson -p path/to/labeled_data.csv -o datasets/davidson -s "[0.8, 0.1, 0.1]"
```

# Training
Create a config file (see `configs/train/example.json`, which contains the hyperparameters used in the thesis) and run:
```
python -m hs_generalization.train -c configs/train/example.json
```
The training mode is set in the config via `task.train_mode`: one of `3class`, `hate_nonhate`, `nonclean_clean`, `hate_clean`. The size of the classification head is derived from the mode automatically. A checkpoint is saved after each epoch as `seed{seed}_{model_name}_{epoch}.pt`; per the thesis, select the checkpoint with the best validation macro F1 per seed.

# Evaluation
Create a config file (see `configs/test/example.json`) and run like the example:

```
# running based on a single seed/checkpoint
python -m hs_generalization.test -c configs/test/example.json --dataset davidson --eval-mode 3class --train-mode 3class --seed 5 --checkpoint "outputs/davidson/RoBERTa-base/3class/seed5_RoBERTa-base_7.pt"

# evaluating the same checkpoint on HateCheck-XR
python -m hs_generalization.test -c configs/test/example.json --dataset hatecheck_xr --eval-mode 3class --train-mode 3class --seed 5 --checkpoint "outputs/davidson/RoBERTa-base/3class/seed5_RoBERTa-base_7.pt" --hatecheck-csv datasets/hatecheck-xr/hatecheck-xr.csv

# If you want to run multiple seeds/checkpoints at once, use run_many.py:
python -m hs_generalization.run_many ^
  -c configs/test/example.json ^
  --dataset hatecheck_xr ^
  --eval-mode 3class ^
  --train-mode 3class ^
  --seeds 7,222,550,999,3111 ^
  --ckpt-pattern "outputs/davidson/RoBERTa-base/3class/seed{seed}_*.pt" ^
  --hatecheck-csv datasets/hatecheck-xr/hatecheck-xr.csv
```
`--train-mode` describes the label space the checkpoint was trained on; `--eval-mode` describes how the ground truth is scored.

# Update (2026 July):
I transformed my Master's thesis project on hate speech detection into a production-ready content moderation platform. Extended a fine-tuned RoBERTa classifier with a multi-mode REST API (FastAPI), RAG-powered policy explanations using ChromaDB, containerization with Docker, experiment tracking via MLflow, and CI/CD automation with GitHub Actions. The system supports three classification modes—ternary (hateful/offensive/clean), binary hate detection, and toxicity filtering—making it adaptable for different moderation use cases.
Available on [this repository](https://github.com/areumb/moderation).
