# JAM Just Ask for Music - Natura Language-based Recommendations for Pre-trained Multimodal Item Recommendations


## Installation

### Environment

- Install the environment with
  `conda env create -f jam.yml`
- Activate the environment with `conda activate jam`


### Data

- move into the folder of the dataset of your interest
- run the associated notebooks for the pre-processing and splitting
- a folder `processed` should have the following files (/ indicates or):
    - <train/val/test>_split.tsv
    - <user/item>_idxs.tsv
    - <user/item>_<any_modality>_features.npy

### Logging

- JAM uses [W&B](https://wandb.ai/site) for logging. If you plan to use log things, you should create an account there first
- Modify the `constants/wandb_constants.py` file with your `entity_name` and `project_name`


## Usage



## Cite


## License




