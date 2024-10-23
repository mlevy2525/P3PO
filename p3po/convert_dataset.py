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
pickle_path = f"/home/ademi/hermes/data/open_oven_20241022_preprocessed/{task_name}_{dimensions}d_closed_loop_dataset.pkl"

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

pickle.dump(to_return, open(f"/home/ademi/P3PO/expert_demos/general/{task_name}_{dimensions}d_closed_loop_dataset.pkl", "wb"))
pickle.dump(to_return, open(f"/home/ademi/P3PO/processed_data/points/{task_name}_{dimensions}d_closed_loop_dataset.pkl", "wb"))