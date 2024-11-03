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
    parser.add_argument("--subsample", type=int, default=1, help="subsample rate for the dataset")
    args = parser.parse_args()

    # Calculate the effective sampling rate in Hz
    effective_hz = 30 // args.subsample  # Assuming the default is 30 Hz

    # Adjust the dataset name to include the Hz information
    if args.delta_actions:
        old_closed_loop_dataset_name = f'{args.task_name}_{args.dimensions}d_delta_actions_minlength{args.min_length}_closed_loop_dataset.pkl'
        closed_loop_dataset_name = f'{args.task_name}_{args.dimensions}d_delta_actions_minlength{args.min_length}_{effective_hz}hz_closed_loop_dataset.pkl'
    else:
        old_closed_loop_dataset_name = f'{args.task_name}_{args.dimensions}d_abs_actions_minlength{args.min_length}_closed_loop_dataset.pkl'
        closed_loop_dataset_name = f'{args.task_name}_{args.dimensions}d_abs_actions_minlength{args.min_length}_{effective_hz}hz_closed_loop_dataset.pkl'
    pickle_path = f"{args.preprocessed_data_dir}/{old_closed_loop_dataset_name}"

    to_return = {}
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
        print("There are {} episodes in the dataset".format(len(data)))
        actions = []
        all_graphs = []
        demo_nums = []
        for episode_idx in range(len(data)):
            for start_idx in range(args.subsample):
                # Subsample actions and observations
                episode_actions = data[episode_idx]['actions'][start_idx::args.subsample]
                actions.append(episode_actions)
                demo_nums.append(data[episode_idx]['demo_num'])
                
                graphs = []
                for idx in range(start_idx, len(data[episode_idx]['actions']), args.subsample):
                    concatenated = np.concatenate([
                        data[episode_idx]['object_keypoints'][idx].reshape(-1),
                        data[episode_idx]['hand_keypoints'][idx].reshape(-1)
                    ])
                    graphs.append(concatenated)
                all_graphs.append({'graph': graphs})
        
        to_return['actions'] = actions # list of episodes (which themselves are lists of actions)
        to_return['observations'] = all_graphs # list of episodes (which themselves are lists of observations, object and hand keypoints interleaved)
        to_return['demo_num'] = demo_nums
        to_return['subsample'] = args.subsample

    print("There are {} episodes in the subsampled dataset".format(len(to_return['actions'])))

    # Save the subsampled dataset
    output_path_expert = f"/home/ademi/P3PO/expert_demos/general/{closed_loop_dataset_name}"
    output_path_processed = f"/home/ademi/P3PO/processed_data/points/{closed_loop_dataset_name}"

    with open(output_path_expert, "wb") as f_expert, open(output_path_processed, "wb") as f_processed:
        pickle.dump(to_return, f_expert)
        pickle.dump(to_return, f_processed)

    print(f"Dumped closed loop dataset for {closed_loop_dataset_name} to expert_demos and processed_data folders")