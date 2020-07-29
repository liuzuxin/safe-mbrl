Model-based RL with Constrained Cross Entropy Method
==================================

## Supported Platforms

This repo has been tested on Ubuntu 18.04 LTS, and is probably fine for most recent Mac and Linux operating systems. 

## Setup
### Simulation Environment Installation 

Our experiment environments are modified from [Safety Gym](https://github.com/openai/safety-gym), which depends heavily on [mujoco_py](https://github.com/openai/mujoco-py). So the first step is installing [MuJoCo-200](https://www.roboti.us/index.html): download binaries, put license file inside, and add path to `~/.bashrc`. See the [mujoco_py](https://github.com/openai/mujoco-py) documentation for details. Note that mujoco_py **requires Python 3.6 or greater**, so our simulation environments do as well.

Afterwards, simply install our modified Safety Gym environments by:

```
cd env

pip install -e .
```

### Other Requirements
- PyTorch-1.4.0
- gym-0.15
- CUDA-10.0 (recommended if you want GPU acceleration)
- CUDNN-7.6 (recommended if you want GPU acceleration)

To installing other dependencies (tqdm, pyyaml, mpi4py, psutil, matplotlib, seaborn), simply run:
  ```Shell
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

### LightGBM installation

Our method depends one [LightGBM](https://lightgbm.readthedocs.io/en/latest/) model, we recommend to install LightGBM package through pip or conda:

```
pip install lightgbm
```
or
```
conda install -c conda-forge lightgbm
```
We suggest you install pytorch first as there might be package conflicts if installing lightgbm first.

If you want to install GPU-version of LightGBM, please refer to their [documentation](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html)


## Usage

  
### Train with Default Configuration
- Train agent with MPC + CCE + TS + Model ensemble in the PointGoal1 environment:
```Shell
python run.py --robot point --task goal --level 1 --ensemble 4 --config ./data/config.yml
```

To plot data:

```
python utils/plot.py data/test
```

### Arguments and Parameters
| Flags and Parameters  | Description |
| ------------- | ------------- |
| ``--robot``  | robot model, selected from `point` or `car`  |
| ``--task``  | task, selected from `goal` or `button` |
| ``--level``  | environment difficulty, selected from `1` or `2`, where `2` would be more difficult than `1`  |
| ``--epoch``  | maximum epochs to train  |
| ``--episode``  | determines how many episodes data to collect for each epoch  |
| ``--render``  | render the environment |
| ``--test``  | test the performance of pretrained models without training  |
| ``--seed``  | (Optional) Seed for Gym, PyTorch and Numpy  |
| ``--dir``  |  directory to save the logging information  |
| ``--name``  | name of the experiment, used to save data in a folder named by this parameter  |
| ``--save``  | save the trained dynamic model, data buffer, and cost model  |
| ``--load``  | load the trained dynamic model, data buffer, and cost model from a specified directory  |
| ``--ensemble``  | number of model ensembles  |
| ``--config``  | specify the path to the configuation file of the models  |
