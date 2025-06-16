# Dorado Compression 

This repository contains the main functional code towards the completion of the final year report. The workflows are presented in the form of `.ipynb` notebooks. 

Note that due to vscode remote compatibility issues, many code cells are instead used to produce bash scripts to be executed in a separate terminal on a GPU machine. The order which the scripts need to be run to reproduce the results should be largely preserved by the order narrated in the notebooks.

The full project working directory with python 3.10 virtual environments can be found under `/vol/bitbucket/bl1821/` on DoC server, which also contains:
- Full basecalling/alignment experimental data and the trained models for the project under `submission/`
- Code for many previously attempted approaches under `legacy/` with some historical experiment logs.

## Repository structure

The `creature` directory contains the code used in the workflow described in Section 3 of the final report. Some parameters in the workflow files need to be carefully adjusted to correctly produce the results due to the amount of exploration and reworks during this part of the project.

The `frankenstein` directory contains the code used to reproduce the exact workflow described in Section 4 of the final report. The contents under this directory are relatively well-organised and is designed to replicate the results.

The `shortcut.sh` contains some of the environment functions and variables used in the project, which are sometimes quoted in the workflow notebooks.

The use of `train.py` is described in the next section.

## Bonito adaptations

The modifications of the Bonito repository for this project include custom training and exporting pipelines. However, the forked repository has not been implemented clean enough due to the time constraints and may be difficult to use. Therefore, the main changes are sorted into the `train.py` file directly under this repository.

To effect the changes, install `ont-bonito==0.9.0` and replace `bonito.cli.train` with the `train.py` from this directory, which contains the custom `Trainer` class adapted from `bonito.training` and additional code for masking.

## Declaration of originality

The `convolutional.py` and `encode.py` within several locations of the repository are cloned from [DnaModelCompression](https://github.com/Omer-Sella/DnaModelCompression/tree/main) and [TurboDNA](https://github.com/Omer-Sella/turboDNA) respectively for early data preparation. The technology of [ChatGPT 4.0](https://openai.com/chatgpt/overview/) has been used for interpreting error messages and adding debug outputs. The design of the workflow, code writing and experiment execution are of my own work.
