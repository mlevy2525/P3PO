# P3-PO: Prescriptive Point Priors for Visuo-Spatial Generalization of Robot Policies

## Installation
The first thing you'll need to do is set up the conda environment and download the submodules.

- Clone the repository and create a conda environment using the provided `conda_env.yml` file.
```
conda env create -f conda_env.yml
```
- Activate the environment using `conda activate p3-po`. Please note that different computers and environments will require different installations for pytorch and the cudatoolkit. We install them with pip install, but make sure that this is compatible with your computer's version of cuda. If it is not you should be fine to just uninstall pytorch and download it as makes sense for your set up.

- Download the submodules with the following command
```
git submodule update --init --recursive
```

- You can install these submodules and relevant packages by running the setup.sh file. Make sure to run this from the root repository or the models may get installed in the wrong location.
```
./setup.sh
```

## Labeling the points
- The next thing you will need to do is label your "prescriptive points". We have included a jupyter notebook to do this in the P3PO/p3po/data_generation folder.

- Open the label_points notebook. In the first cell you will need to set several variables.

- The first is the path to the image/video you want to label points on. If you want to label an image make sure you set use_video to False.

- Next you'll need to name your task, remember this name as you'll need to use it later to point the final code towards your labeled points.

- If the image is too small to label accurate points you can set size_multiplier to a larger number, this won't change the final points.

- Once you have set these you can run the all the cells and label points at the bottom. You can do this by clicking on the image. Once you are done please select the Save Images button.

- We have included an image (#TODO: ADD IMAGE HERE) to show you an example of what labeled points may look like. 

- If you find that the tracking is not as good as you would like you can label some additional points on each object as shown below (#TODO: ADD IMAGE HERE). This will likely improve cotrackers accuracy. Make sure to label these points AFTER you have selected all of the prescriptive points and also you will need to set the number of prescriptive points in the next step.

## Labeling the data
- Now that you have selected your points you can generate your dataset using the generate_points.py script.

- Open the generate_points.py script and finish the 3 TODOs labeled there. Additionally, make sure to set your path and task name in the config.yaml file. If you labeled additional points in the prior step you will need to set num_points here. If not you can leave this set to -1.

```
python generate_points.py
```




