import pickle
import numpy as np

pickle_path = "/home/ademi/hermes/data/push_donut_franka_20241014_preprocessed/closed_loop_dataset_objectpixelkeypoints.pkl"

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

pickle.dump(to_return, open("/home/ademi/P3PO/expert_demos/general/closed_loop_dataset_objectpixelkeypoints.pkl", "wb"))
pickle.dump(to_return, open("/home/ademi/P3PO/processed_data/points/closed_loop_dataset_objectpixelkeypoints.pkl", "wb"))