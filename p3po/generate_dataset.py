import pickle 
import numpy as np
import os
import h5py

from hermes.utils.data_utils import map_aruco_to_dmanus


def _load_files(demo_path):
    # Load the indices files 
    image_indices_path = os.path.join(
        demo_path, "image_indices_cam_{}.pkl".format(3)
    )
    with open(image_indices_path, "rb") as file:
        image_indices = pickle.load(file)

    hand_pose_indices_path = os.path.join(
        demo_path, 'fingertips.pkl'
    )
    hand_poses_path = os.path.join(
        demo_path, 'fingertips.h5'
    )
    with open(hand_pose_indices_path, 'rb') as file: 
        hand_pose_indices = pickle.load(file)

    with h5py.File(hand_poses_path, "r") as file:
        fingertips_wrt_world = np.asarray(file['fingertips'])

    # Load the aruco
    aruco_pose_path = os.path.join(demo_path, 'aruco_postprocessed.npz')
    aruco_poses = np.load(aruco_pose_path)
    return image_indices, fingertips_wrt_world, hand_pose_indices, aruco_poses


def get_fingertips_in_camera_frame(preprocessed_data_dir):
    image_indices, fingertips_wrt_world, hand_pose_indices, aruco_poses = _load_files(preprocessed_data_dir)
    hand_trajectory = []
    for data_id in range(len(image_indices)):
        _, image_frame_id = image_indices[data_id]
        aruco_id = image_frame_id - aruco_poses['indices'][0]
        _, fingertip_id = hand_pose_indices[data_id]
        H_A_C = aruco_poses['poses'][aruco_id]
        H_A_C = np.concatenate([H_A_C, np.array([[0,0,0,1]])], axis=0)
        H_D_C = map_aruco_to_dmanus(H_A_C) # dmanus-to-camera
        
        fingertips_in_world = fingertips_wrt_world[fingertip_id]  
        H_F_Cs = []      
        for i,finger_tvec in enumerate(fingertips_in_world):
            # Create the homo for finger_tvec
            H_F_D = np.eye(4) # H_F_D: fingertip-to-dmanus
            H_F_D[:3, 3] = finger_tvec[:]

            # fingertip-to-camera
            H_F_C = H_D_C @ H_F_D
            finger_tvec_in_camera_frame = H_F_C[:3, 3]
            H_F_Cs.append(finger_tvec_in_camera_frame)
        
        H_F_Cs = np.stack(H_F_Cs, axis=0)
        hand_trajectory.append(H_F_Cs)
    return hand_trajectory


def get_object_keypoints_in_camera_frame_videowisecotracker(trajectory, preprocessed_data_dir, use_pixel_keypoints):
    image_indices, _, _, _ = _load_files(preprocessed_data_dir)
    ''' assumes origin is top left of the image with r left to right and c top to bottom'''
    
    trajectory_camera_coords = []
    
    for data_id in range(len(image_indices)):
        _, image_frame_id = image_indices[data_id]
        frame_camera_coords = []
        for coords in trajectory[image_frame_id]:
            if use_pixel_keypoints:
                for i in range(len(coords) // 2):
                    frame_camera_coords.append([coords[2*i], coords[2*i+1]])
            else:
                for i in range(len(coords) // 3):
                    frame_camera_coords.append([coords[3*i], coords[3*i+1], coords[3*i+2]])
        trajectory_camera_coords.append(np.array(frame_camera_coords))
    
    return trajectory_camera_coords


def get_object_keypoints_in_camera_frame_framewisecotracker(trajectory, use_pixel_keypoints):
    ''' assumes origin is top left of the image with r left to right and c top to bottom'''
    
    trajectory_camera_coords = []
    
    for frame_idx in range(len(trajectory)):
        frame_camera_coords = []
        for coords in trajectory[frame_idx]:
            if use_pixel_keypoints:
                for i in range(len(coords) // 2):
                    frame_camera_coords.append([coords[2*i], coords[2*i+1]])
            else:
                for i in range(len(coords) // 3):
                    frame_camera_coords.append([coords[3*i], coords[3*i+1], coords[3*i+2]])
        trajectory_camera_coords.append(np.array(frame_camera_coords))
    
    return trajectory_camera_coords


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_data_dir", type=str, required=True, help="path to demo data folder with preprocessed outputs")
    parser.add_argument("--task_name", type=str, required=True, help="name of the task")
    parser.add_argument("--dimensions", type=int, default=3, help="number of dimensions to use for the object keypoints")
    parser.add_argument("--delta_actions", default=False, action="store_true", help="whether to use delta actions or not")
    parser.add_argument("--min_length", type=int, default=100, help="minimum length of the trajectory")
    parser.add_argument("--framewise_cotracker", default=False, action="store_true", help="whether used frame-wise cotracker or not")
    parser.add_argument("--subsample", type=int, default=3, help="subsample rate for the dataset")
    parser.add_argument("--unproject_depth", default=False, action="store_true", help="Whether to unproject depth to compute keypoints")
    args = parser.parse_args()

    missing_files = 0
    too_short = 0
    point_type = '3d' if args.unproject_depth else '2.5d'

    dataset = []
    if args.framewise_cotracker:
        object_keypoints_file = args.preprocessed_data_dir + f"/{args.task_name}_{point_type}_cov3_framewise.pkl"
    else:
        object_keypoints_file = args.preprocessed_data_dir + f"/{args.task_name}_{point_type}_videowise.pkl"
    with open(object_keypoints_file, "rb") as file:
        object_keypoints = pickle.load(file)
    traj_lengths = []
    for demo_num, object_keypoint_trajectory in object_keypoints.items():
        hand_trajectory_path = args.preprocessed_data_dir + f"/demonstration_{demo_num}"
        if f"demonstration_{demo_num}" not in os.listdir(args.preprocessed_data_dir) or not os.path.exists(hand_trajectory_path + "/image_indices_cam_3.pkl"):
            print(f"Skipping {hand_trajectory_path} because it does not exist or doesn't have the necessary files")
            missing_files += 1
            continue
        if args.framewise_cotracker:
            H_O_C_trajectory = get_object_keypoints_in_camera_frame_framewisecotracker(trajectory=object_keypoint_trajectory, use_pixel_keypoints=args.dimensions==2)
        else:
            H_O_C_trajectory = get_object_keypoints_in_camera_frame_videowisecotracker(trajectory=object_keypoint_trajectory, preprocessed_data_dir=hand_trajectory_path, use_pixel_keypoints=args.dimensions==2)
        H_F_C_trajectory = get_fingertips_in_camera_frame(preprocessed_data_dir=hand_trajectory_path)
        assert len(H_O_C_trajectory) == len(H_F_C_trajectory)
        traj_lengths.append(len(H_F_C_trajectory))
        if len(H_F_C_trajectory) < args.min_length:
            print(f"Skipping {hand_trajectory_path} because it has less than {args.min_length} frames")
            too_short += 1
            continue
        demonstration = {'object_keypoints': H_O_C_trajectory, 'hand_keypoints': H_F_C_trajectory, 'demo_num': demo_num}
        dataset.append(demonstration)
    for demo in dataset:
        if args.delta_actions:
            demo['actions'] = [demo['hand_keypoints'][i+1] - demo['hand_keypoints'][i] for i in range(len(demo['hand_keypoints'])-1)]
        else:
            demo['actions'] = [demo['hand_keypoints'][i+1] for i in range(len(demo['hand_keypoints'])-1)]

    dataset = sorted(dataset, key=lambda x: x['demo_num'])

    print(f"Generated dataset contains {len(dataset)} demonstrations")
    print(f"This excludes... Number of demonstrations skipped due to missing files: {missing_files}")
    print(f"This excludes... Number of demonstrations skipped due to being too short: {too_short}")
    print(f"Shape of the object_keypoints: {dataset[0]['object_keypoints'][0].shape}")
    print(f"Shape of the hand_keypoints: {dataset[0]['hand_keypoints'][0].shape}")
    print(f"Shape of the actions: {dataset[0]['actions'][0].shape}")
    print("Average length of object keypoint demonstrations: ", np.mean([len(demo['object_keypoints']) for demo in dataset]))
    print("Average length of hand keypoint demonstrations: ", np.mean([len(demo['hand_keypoints']) for demo in dataset]))
    print("Average length of action sequence: ", np.mean([len(demo['actions']) for demo in dataset]))

    if args.delta_actions:
        dataset_name = f"{args.task_name}_{point_type}_delta_actions_minlength{args.min_length}_closed_loop_dataset.pkl"
    else:
        dataset_name = f"{args.task_name}_{point_type}_abs_actions_minlength{args.min_length}_closed_loop_dataset.pkl"

    with open(args.preprocessed_data_dir + f"/{dataset_name}", "wb") as file: 
        pickle.dump(dataset, file)
    print(f"Dataset saved to {args.preprocessed_data_dir}/{dataset_name}")

    import matplotlib.pyplot as plt
    if not os.path.exists("dataset_plots"):
        os.makedirs("dataset_plots")

    plt.hist(traj_lengths, bins=20)
    plt.xlabel("Trajectory Length")
    plt.ylabel("Frequency")
    plt.title("Histogram of Trajectory Lengths")
    plt.savefig(f"dataset_plots/histogram_of_trajectory_lengths.png")

    if args.dimensions == 3:
        colors = ['r', 'g', 'b', 'c']
        for demo_num in range(10):
            fingertips_3d = np.array(dataset[demo_num]['hand_keypoints'])
            object_keypoints_3d = dataset[demo_num]['object_keypoints']
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for i in range(4):
                ax.plot(fingertips_3d[:, i, 0], fingertips_3d[:, i, 1], fingertips_3d[:, i, 2], label=f'Fingertip {i}', color=colors[i])
            ax.scatter(fingertips_3d[0, :, 0], fingertips_3d[0, :, 1], fingertips_3d[0, :, 2], color='green', marker='o', s=100, label='Start')
            ax.scatter(fingertips_3d[-1, :, 0], fingertips_3d[-1, :, 1], fingertips_3d[-1, :, 2], color='red', marker='*', s=100, label='Finish')
            object_keypoint_x_trajectory = [object_keypoints_3d[i][-1][0] for i in range(len(object_keypoints_3d))]
            object_keypoint_y_trajectory = [object_keypoints_3d[i][-1][1] for i in range(len(object_keypoints_3d))]
            object_keypoint_z_trajectory = [object_keypoints_3d[i][-1][2] for i in range(len(object_keypoints_3d))]
            ax.scatter(object_keypoint_x_trajectory, object_keypoint_y_trajectory, object_keypoint_z_trajectory, label='Object keypoints', color='purple', marker='x')
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')
            ax.legend()
            ax.set_xlim([-0.5, 0.5])
            ax.set_ylim([0, 1.0])
            ax.set_zlim([0, 1.0])
            plt.savefig(f"dataset_plots/hand_trajectory_demo_{demo_num}.png")
            print(f"Hand trajectory plot saved to hand_trajectory_demo_{demo_num}.png")
            plt.close()
    elif args.dimensions == 2:
        colors = ['r', 'g', 'b', 'c']
        for demo_num in range(10):
            fingertips_2d = np.load(f'/data/hermes/push_donut_franka_20241014_preprocessed/demonstration_{demo_num}/2d_fingertips.npy')
            object_keypoints_2d = dataset[demo_num]['object_keypoints']
            fig, ax = plt.subplots()
            for i in range(4):
                ax.plot(fingertips_2d[:, i, 0], fingertips_2d[:, i, 1], label=f'Fingertip {i}', color=colors[i])
            ax.scatter(fingertips_2d[0, :, 0], fingertips_2d[0, :, 1], color='green', marker='o', s=100, label='Start')
            ax.scatter(fingertips_2d[-1, :, 0], fingertips_2d[-1, :, 1], color='red', marker='*', s=100, label='Finish')
            object_keypoint_x_trajectory = [object_keypoints_2d[i][0][0] for i in range(len(object_keypoints_2d))]
            object_keypoint_y_trajectory = [object_keypoints_2d[i][0][1] for i in range(len(object_keypoints_2d))]
            ax.plot(object_keypoint_x_trajectory, object_keypoint_y_trajectory, label='Object keypoints', color='purple', marker='x')
            ax.legend()
            ax.set_xlim([500, 1200])
            ax.set_ylim([350, 600])
            plt.savefig(f"dataset_plots/hand_trajectory_demo_{demo_num}.png")
            print(f"Hand trajectory plot saved to hand_trajectory_demo_{demo_num}.png")
            plt.close()
    else:
        raise ValueError("Dimensions must be either 2 or 3")
    
    # Calculate the effective sampling rate in Hz
    effective_hz = 30 // args.subsample  # Assuming the default is 30 Hz

    # Adjust the dataset name to include the Hz information
    if args.delta_actions:
        old_closed_loop_dataset_name = f'{args.task_name}_{point_type}_delta_actions_minlength{args.min_length}_closed_loop_dataset.pkl'
        closed_loop_dataset_name = f'{args.task_name}_{point_type}_delta_actions_minlength{args.min_length}_{effective_hz}hz_closed_loop_dataset.pkl'
    else:
        old_closed_loop_dataset_name = f'{args.task_name}_{point_type}_abs_actions_minlength{args.min_length}_closed_loop_dataset.pkl'
        closed_loop_dataset_name = f'{args.task_name}_{point_type}_abs_actions_minlength{args.min_length}_{effective_hz}hz_closed_loop_dataset.pkl'
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
