import pickle
import numpy as np
import yaml

with open("cfgs/suite/p3po.yaml") as stream:
    try:
        cfg = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

task_name = cfg['task_name']
dimensions = cfg['dimensions']
delta_actions = False
if delta_actions:
    closed_loop_dataset_name = f'{task_name}_{dimensions}d_delta_actions_closed_loop_dataset.pkl'
else:
    closed_loop_dataset_name = f'{task_name}_{dimensions}d_abs_actions_closed_loop_dataset.pkl'
pickle_path = f"/home/ademi/hermes/data/open_oven_20241022_preprocessed/{closed_loop_dataset_name}"

to_return = {}
with open(pickle_path, "rb") as f:
    data = pickle.load(f)
    actions = []
    all_graphs = []
    for episode_idx in range(len(data)):
        actions.append(data[episode_idx]['actions'])
        graphs = []
        for idx in range(len(data[episode_idx]['actions'])):
            concatenated = np.concatenate([data[episode_idx]['object_keypoints'][idx].reshape(-1), data[episode_idx]['hand_keypoints'][idx].reshape(-1)])
            graphs.append(concatenated)
        all_graphs.append({'graph': graphs})
    
    to_return['actions'] = actions
    to_return['observations'] = all_graphs

pickle.dump(to_return, open(f"/home/ademi/P3PO/expert_demos/general/{closed_loop_dataset_name}", "wb"))
pickle.dump(to_return, open(f"/home/ademi/P3PO/processed_data/points/{closed_loop_dataset_name}", "wb"))
print(f"Dumped closed loop dataset for {closed_loop_dataset_name} to expert_demos and processed_data folders")