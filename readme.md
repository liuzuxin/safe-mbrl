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

Our method depends on [LightGBM](https://lightgbm.readthedocs.io/en/latest/) model; we recommend to install LightGBM package through pip or conda:

```
pip install lightgbm
```
or
```
conda install -c conda-forge lightgbm
```
We suggest you install pytorch first as there might be package conflicts if installing lightgbm first.

If you want to install GPU-version of LightGBM, please refer to their [documentation](https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html).


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

#### Arguments and Parameters
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


### Plot a Single Figure from Data
To plot a single figure:
```
python script/plot.py data/test
```
![image](https://github.com/liuzuxin/safe-mbrl/tree/master/data/TestFigure1.png)
```
python script/plot.py data/pg1/ensemble-rce data/pg1/ensemble-cem --hline 14 15 --linename Test1 Test2
```
![image](https://github.com/liuzuxin/safe-mbrl/tree/master/data/TestFigure3.png)

Horizontal lines can be used as convergence values for model-free methods, as recalled from the proposed paper.

The script does not yet support reading a combination of model-free and model-based data as their data files are coded differently. Be careful when selecing the paths to files.

#### Arguments and Parameters
| Flags and Parameters  | Description |
| ------------- | ------------- |
| Mandatory argument  | list of paths to data `progress.txt`; all sub-directories of the paths will be scanned.  |
| ``--xaxis``  | the data that will be plotted as the x-axis. e.g. `TotalEnvInteracts`, `Episode`  |
| ``--yaxis``  | the data that will be plotted as the y-axis. e.g. `EpRet` is the reward in 1 episode; `EpCost` is the cost in 1 episode  |
| ``--condition``  | how to categorize the plotted lines; select `Method` to group data by method of experiment, `BySeed` to separate individual trials  |
| ``--smooth``  | determines how smoothening is done while plotting; larger value means more smoothening; default `50`; input `0` to turn off  |
| ``--cut``  | determines how to shorten the datasets for alignment; select `2` for no action, `1` to align each group of data, `0` for global shortest alignment |
| ``--hline``  | (Optional) the y coordinates where horizontal dotted lines will be plotted; input a list of numbers |
| ``--linename``  | (Optional) a list of strings that are the labels to the above horizontal lines, respectively  |


### Average and Aggregate from Data
As used in Table 1 of the proposed paper, mean and sum of data can be taken using the following method:
```
python script/count.py data/cg1/ensemble-rce data/cg1/ensemble-cem --sum 100
```
Mean value of the targetted label is taken across a group; e.g. mean `cost` for `RCE method`. The first `N` mean values are then summed for each group.
The output format follows: {Group name: Sum of N mean values}

#### Arguments and Parameters
| Flags and Parameters  | Description |
| ------------- | ------------- |
| Mandatory argument  | list of paths to data `progress.txt`; all sub-directories of the paths will be scanned.  |
| ``--target``  | the targetted label. e.g. `EpRet`, `EpCost`  |
| ``--condition``  | how to categorize the plotted lines; select `Method` to group data by method of experiment, `BySeed` to separate individual trials  |
| ``--cut``  | determines how to shorten the datasets for alignment; select `2` for no action, `1` to align each group of data, `0` for global shortest alignment |
| ``--sum``  | `N`, sum the first `N` elements  |


