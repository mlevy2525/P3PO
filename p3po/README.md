# P3PO // Hermes

After running all the preprocessing commands in `hermes`, continue preprocessing the data for training Baku model in P3PO
```bash
cd p3po/

# Set the following variables
export TASKNAME="collect_ball_20241106c"

# In the first cell, modify the following variables:
#   - path: to point to a segmented mp4 video in the preprocessed folder for a single demonstration
#   - task_name: the name that you want this task to be called throughout the rest of the pipeline
# Then, on the image, click all the prescriptive points for the model
ipython label_points.ipynb

# Run cotracker on the entire dataset (this will take about 1 min/iter)
cd data_generation/
python generate_points_framewise.py --preprocessed_data_dir "/home/ademi/hermes/data/${TASKNAME}_preprocessed" --task_name $TASKNAME --num_tracked_points 8 --keypoints_type 3

# To run things in batch mode
# bash batch_generate_train_dataset.sh

# Generate dataset (this should be quick)
cd ../
python generate_dataset.py --preprocessed_data_dir "/home/ademi/hermes/data/${TASKNAME}_preprocessed" --task_name $TASKNAME --history_length 10 --subsample 3 --framewise_cotracker --delta_actions --keypoints_type 3

# Train (YOU NEED TO CHANGE THE HARD-CODED PARAMS IN run_train.sh)
bash run_train.sh

# To run things in batch mode
# bash batch_run_train.sh
```
