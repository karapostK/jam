# :jar: JAM - Just Ask for Music  
### Natural Language-based Recommendations for Pre-trained Multimodal Item Recommendations


<div align="center">
    <img src="./assets/jam_cute.png" style="width: 320px" />
</div>

## Installation & Setup

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

- JAM uses [W&B](https://wandb.ai/site) for logging. If you plan to log things, you should create an account there first
- Modify the `constants/wandb_constants.py` file with your `entity_name` and `project_name`
- First time usage you might want call `wandb login`.


## Usage
General flow is
1. Create a configuration file
2. Call `run_experiments.py`

The framework will take care of:
- Loading the data
- Training/Validating (optionally Testing) the model
- Saving the best model and configuration
- Log results to W&B
### Running a Single Experiment
A single experiments can be 1) `train/val` + `test` 2) `train/val` 3) just `test`

1. Create a `.yml` config file (possibly in `conf/confs/`). See `conf/confs/template_conf.yml` for explanations of the possible values. See `constants/conf_constants.py` for defaults.
   1. Minimally, you should specify `data_path`, where `data/<dataset_name>` is looked for.
   2. Additionally, you should also add hyperparameters of your chosen algorithm.
   3. Running `test` as experiment type requires `model_path` to the saved model.
2. `python run_experiments.py -a <alg> -d <dataset> -c <path_to_conf> -t <run_type>`
   1. For `alg` and `dataset` see the available ones in `constants/enums.py`
   2. `path_to_conf` is what you specified above
   3. For `run_type` and other variables see `run_experiments.py`
3. Look at how your experiment is doing on W&B.

Example:
`test_conf.yml`
```yml
data_path: "/home/PycharmProjects/jam/data"
d: 28 # for avgmatching model
device: cuda

n_epochs: 50
eval_batch_size: 256
train_batch_size: 256

running_settings:
  train_n_workers: 4
  eval_n_workers: 4
  batch_verbose: 1
```
then run
`python run_experiment.py -a basematching -d amazon23office -c conf/confs/test_conf.yml`
(if `-t` is not specified, it will run `train/val/test`)


### Running Multiple Experiments (Sweeps with W&B)
To run multiple experiments, JAM relies on W&B sweeps. This is to execute different `train/val` experiments.

1. Create a `.yml` config file (possibly in `conf/sweeps/`). Take `conf/sweeps/template_sweep_conf.yml` as reference. See `constants/conf_constants.py` for defaults.
      1. Specify again your `entity_name` and `project_name` in the conf. These are the same values you had for the Logging step above.
      2. Give a meaningful name to your sweep (e.g. `<algorithm_name>-<dataset_name>` should suffice. Add these values to the `parameters` section as well. See `constants/enums.py` for possible values. 
      3. Adjust the rest of the configuration as you please. See the [official docs](https://docs.wandb.ai/guides/sweeps/) on W&B.  
2. Activate the sweep. `wandb sweep conf/sweeps/test_sweep_conf.yml`. Your sweep now should be online and can be monitored on your dashboard.
3. Start 1+ agents. The command to start an agent is returned by wandb when activating the sweep. It's in the shape of `wandb agent <entity_name>/<project_name>/<sweep_id>`.

NB. You can adjust how many gpus are visible to the agent by specifying `CUDA_VISIBLE_DEVICES=... wandb agent..`

#### Multiple Runs on a Server
When you activated a sweep, and you don't want to start each single agent, you can use `run_agents.py`. Originally from my cool colleague [Christian](https://github.com/Tigxy/SiBraR---Single-Branch-Recommender/blob/main/run_agent.py)

When running `run_agent.py` you need to specify:
- `sweep_id`. This value should be in the format `<entity_name>/<project_name>/<sweep_id>`
- `available_gpus` (e.g. the indexes)
- `n_parallel` or how many agents PER gpu. Need to be careful with also the # of workers.
## Extend
### Codebase Structure
```
.
├── algorithms                  <- Classes about Query-User-Item Matching
├── conf                        <- Parsing & Storing .yml conf file
├── constants                   <- Constants & Enums used across the codebase
├── data                        <- Data classes, Raw and Processed Datasets
├── evaluation                  <- Metrics and Evaluation Procedure
├── train                       <- Trainer class
├── utilities                   <- Utilities of mild to low importance
├── (saved_models)              <- Automatically created (if def. conf is not altered)
├── experiment_helper.py        <- Executes the main functionalities of the code.
└── run_experiment.py           <- Entry point to the code
```
### Add Algorithms
Take a look at the `BaseQueryMatchingModel`in `algorithms/base` on what functionalities are expected from a new algorithm.

You can implement your class in `algorithms/alg` (e.g. look at `AverageQueryMatching`). Creating a descendent of `BaseQueryMatchingModel` would be the best ;). 

When the main methods are implemented, add your class to `AlgorithmsEnum` in `constants/enums.py` so it can be recognized when calling `run_experiments`

### Add Datasets
Choose a name, short and lowercase letters to denote the dataset `<dataset_name>`

The expected format of the files are in the first lines in `data/datasets.py` (for the user-query-item matching) and `data/feature.py` (for pre-trained user/item features).

If you can provide the files in the above format, you can add them to ``data/<dataset_name>/processed`

Add your dataset to `DatasetsEnum` in `constants/enums.py`.

NB. Codebase will look for the data in `os.path.join(conf['data_path'],<dataset_name>,'processed')`

## Cite


## License




