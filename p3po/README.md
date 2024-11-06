# P3PO // Hermes

After running all the preprocessing commands in `hermes`, continue preprocessing the data for training Baku model in P3PO
```bash
cd p3po/

# Set the following variables
export PREPROCESSDIR="/home/ademi/hermes/data/pick_block_20241103a_preprocessed_30hz"
export TASKNAME="pick_block_20241103a_30hz_nooccpoints"
# export PREPROCESSDIR="/home/ademi/hermes/data/open_drawer_20241103b_preprocessed_30hz"
# export TASKNAME="open_drawer_20241103b_30hz"

# In the first cell, modify the following variables:
#   - path: to point to a segmented mp4 video in the preprocessed folder for a single demonstration
#   - task_name: the name that you want this task to be called throughout the rest of the pipeline
# Then, on the image, click all the prescriptive points for the model
ipython label_points.ipynb

# Run cotracker on the entire dataset (this will take about 1 min/iter)
cd data_generation/
python generate_points_framewise.py --preprocessed_data_dir $PREPROCESSDIR --task_name $TASKNAME --num_tracked_points 8 --unproject_depth

# Generate dataset (this should be quick)
cd ../
python generate_dataset.py --preprocessed_data_dir $PREPROCESSDIR --task_name $TASKNAME --min_length 33 --subsample 3 --framewise_cotracker --unproject_depth

# Train (YOU NEED TO CHANGE THE HARD-CODED PARAMS IN run_train.sh)
bash run_train.sh
```
