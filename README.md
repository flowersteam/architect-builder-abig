# Code for Learning to Guide and to Be Guided in the Architect-Builder Problem


## Prerequisites
Install `alfred`, `env_comem` and `main_comem` by going into the corresponding folders and following the dedicate `README.md` files.

To make using alfred as seamless as possible, add the followings to your `.bachrc`:
```
alias alprep='python -m alfred.prepare_schedule'
alias allaunch='python -m alfred.launch_schedule'
alias alclean='python -m alfred.clean_interrupted'
alias alplot='python -m alfred.make_plot_arrays'
alias alretrain='python -m alfred.create_retrainbest'
alias albench='python -m alfred.benchmark'
alias alsync='python -m alfred.sync_wandb'
alias alcopy='python -m alfred.copy_config'
alias alupdate='python -m alfred.update_config_unique'
```
## Short description
* `alfred` is used to manage experiments.
* `env_comem` contains the code for BuildWorld.
* `main_comem` contains the code for the algorithms (ABIG).

## Running the experiments

All command lines should be ran from `main_comem/main_comem`.
Each run takes approximately 48h to run on a CPU. 

### Reproducing results with 3 blocks
Prepare the experiments to be ran
```
alprep --desc 3b --schedule_file schedules/bw_task4_bc/grid_schedule_task_4.py
```

Run the experiments (recommended doing in parallel since each run takes 48h, by playing with `allaunch` arguments `--n_processes` and `--n_experiments_per_proc`)
```
allaunch --from_file schedules/bw_task4_bc/list_searches_bw_task4_bc.txt
```

Evaluate the performances
```
python utils/analyses_OOD.py --from_file schedules/bw_task4_bc/grid_schedule_task_4.py
```

Plot the results
```
python utils/make_analyse_OOD.plot --from_file schedules/bw_task4_bc/grid_schedule_task_4.py
```

### Reproducing results with 6 blocks
Prepare the experiments to be ran
```
alprep --desc 6b --schedule_file schedules/bw_task4_bc_6b/grid_schedule_task_4.py
```

Run the experiments (recommended doing in parallel since each run takes 48h, by playing with `allaunch` arguments `--n_processes` and `--n_experiments_per_proc`)
```
allaunch --from_file schedules/bw_task4_bc_6b/list_searches_bw_task4_bc.txt
```

Evaluate the performances
```
python utils/analyses_OOD.py --from_file schedules/bw_task4_bc_6b/grid_schedule_task_4.py
```

Plot the results
```
python utils/make_analyse_OOD.plot --from_file schedules/bw_task4_bc_6b/grid_schedule_task_4.py
```

### Citation

```
@inproceedings{
  barde2022learning,
  title={Learning to Guide and to be Guided in the Architect-Builder Problem},
  author={Paul Barde and Tristan Karch and Derek Nowrouzezahrai and Cl{\'e}ment Moulin-Frier and Christopher Pal and Pierre-Yves Oudeyer},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=swiyAeGzFhQ}
}
```
