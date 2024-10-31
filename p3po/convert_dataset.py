import pickle
import numpy as np
import yaml

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_data_dir", type=str, required=True, help="path to demo data folder with preprocessed outputs")
    parser.add_argument("--task_name", type=str, required=True, help="name of the task")
    parser.add_argument("--dimensions", type=int, default=3, help="number of dimensions to use for the object keypoints")
    parser.add_argument("--delta_actions", type=bool, default=False, help="whether to use delta actions or not")
    parser.add_argument("--min_length", type=int, default=100, help="minimum length of the trajectory")
    args = parser.parse_args()

    if args.delta_actions:
        closed_loop_dataset_name = f'{args.task_name}_{args.dimensions}d_delta_actions_minlength{args.min_length}_closed_loop_dataset.pkl'
    else:
        closed_loop_dataset_name = f'{args.task_name}_{args.dimensions}d_abs_actions_minlength{args.min_length}_closed_loop_dataset.pkl'
    pickle_path = f"{args.preprocessed_data_dir}/{closed_loop_dataset_name}"

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