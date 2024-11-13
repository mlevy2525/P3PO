import pickle 
import numpy as np
import os
import h5py

from data_generation.generate_points_framewise import CAM_ID
from hermes.utils.data_utils import map_aruco_to_dmanus


def _load_files(demo_path):
    # Load the indices files 
    image_indices_path = os.path.join(
        demo_path, "image_indices_cam_{}.pkl".format(0)
    )
    with open(image_indices_path, "rb") as file:
        image_indices = pickle.load(file)

    allegro_commanded_indices_path = allegro_indices_path = os.path.join(
        demo_path, 'allegro_commanded_joint_states.pkl'
    )
    franka_commanded_indices_path = os.path.join(
        demo_path, 'franka_cartesian_states.pkl'
    )

    allegro_indices_path = os.path.join(
        demo_path, 'allegro_joint_states.pkl'
    )
    franka_indices_path = os.path.join(
        demo_path, 'franka_joint_states.pkl'
    )
    allegro_path = os.path.join(
        demo_path, 'allegro_joint_states.h5'
    )
    allegro_commanded_path = os.path.join(
        demo_path, 'allegro_commanded_joint_states.h5'
    )
    franka_path = os.path.join(
        demo_path, 'franka_joint_states.h5'
    )

    franka_commanded_path = os.path.join(
        demo_path, 'franka_cartesian_states.h5'
    )

    with open(allegro_indices_path, 'rb') as file: 
        allegro_indices = pickle.load(file)
    
    with open(franka_indices_path, 'rb') as file: 
        franka_indices = pickle.load(file)

    with open(allegro_commanded_indices_path,'rb') as file:
        allegro_commanded_indices = pickle.load(file)

    with open(franka_commanded_indices_path, 'rb') as file: 
        franka_commanded_indices = pickle.load(file)

    with h5py.File(allegro_path, "r") as file:
        allegro_position = np.asarray(file['positions'])

    with h5py.File(franka_path, "r") as file:
        franka_position = np.asarray(file['positions'])
    
    with h5py.File(allegro_commanded_path, "r") as file:
        allegro_commanded_position = np.asarray(file['positions'])

    with h5py.File(franka_commanded_path, "r") as file:
        position = np.asarray(file['positions'])
        orientation = np.asarray(file['orientations'])
        franka_commanded_position = np.concatenate((position, orientation), axis=1)
    # Load the aruco
    #aruco_pose_path = os.path.join(demo_path, 'aruco_postprocessed.npz')
    #aruco_poses = np.load(aruco_pose_path)
    return image_indices, allegro_position, allegro_indices,franka_position,franka_indices,allegro_commanded_indices,allegro_commanded_position,franka_commanded_indices,franka_commanded_position

'''
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
'''
def get_frankallegro_in_camera_frame(preprocessed_data_dir):
    image_indices, allegro_position, allegro_indices,franka_position,franka_indices,allegro_commanded_indices,allegro_commanded_position,franka_commanded_indices,franka_commanded_position = _load_files(preprocessed_data_dir)
    allegro_franka = []
    commanded_allegro_franka =[]
    for data_id in range(len(image_indices)):
        _, image_frame_id = image_indices[data_id]
        #aruco_id = image_frame_id - aruco_poses['indices'][0]
        _, franka_id = franka_indices[data_id]
        _,allegro_id= allegro_indices[data_id]
        _, franka_commanded_id = franka_commanded_indices[data_id]
        _,allegro_commanded_id= allegro_commanded_indices[data_id]
        franka = franka_position[franka_id]
        allegro = allegro_position[allegro_id]
        franka_c = franka_commanded_position[franka_commanded_id]
        allegro_c = allegro_commanded_position[allegro_commanded_id]
        allegro_franka.append(np.concatenate((franka,allegro))) # (23,) franka(7,) allegro (16,)
        commanded_allegro_franka.append(np.concatenate((franka_c,allegro_c))) # (23,) franka(7,) position+ orientation allegro(16,)
    return allegro_franka,commanded_allegro_franka

def get_object_keypoints_in_camera_frame_videowisecotracker(trajectory, preprocessed_data_dir, use_pixel_keypoints):
    image_indices, _, _, _,_,_,_,_,_ = _load_files(preprocessed_data_dir)
    ''' assumes origin is top left of the image with r left to right and c top to bottom'''
    trajectory_camera_coords = []
    
    for data_id in range(len(image_indices)):
        _, image_frame_id = image_indices[data_id]
        frame_camera_coords = []
        if image_frame_id<len(trajectory):
            for coords in trajectory[image_frame_id]:
                if use_pixel_keypoints:
                    for i in range(len(coords) // 2):
                        frame_camera_coords.append([coords[2*i], coords[2*i+1]])
                else:
                    for i in range(len(coords) // 3):
                        frame_camera_coords.append([coords[3*i], coords[3*i+1], coords[3*i+2]])
        else:
            print(f"{image_frame_id} not in trajectory with length {len(trajectory)} out of index.")
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
    parser.add_argument("--delta_actions", default=False, action="store_true", help="whether to use delta actions or not")
    parser.add_argument("--history_length", type=int, default=10, help="minimum length of the trajectory")
    parser.add_argument("--framewise_cotracker", default=False, action="store_true", help="whether used frame-wise cotracker or not")
    parser.add_argument("--subsample", type=int, default=3, help="subsample rate for the dataset")
    parser.add_argument("--keypoints_type", type=float, choices=[2, 2.5, 3], default=3, help="Whether to unproject depth to compute keypoints")
    args = parser.parse_args()

    args.min_length = (args.history_length + 1) * args.subsample + 1
    args.action_type = "delta" if args.delta_actions else "abs"
    if args.keypoints_type != 2.5:
        args.keypoints_type = int(args.keypoints_type)
    dimensions = 2 if args.keypoints_type == 2 else 3
    print(f"only processing examples that are >={args.min_length} length !!")
    missing_files = 0
    too_short = 0

    object_keypoints_file = args.preprocessed_data_dir + f"/{args.task_name}_{str(args.keypoints_type)}d-keypoints.pkl"  # TODO change this if we ever use videowise
    with open(object_keypoints_file, "rb") as file:
        object_keypoints = pickle.load(file)

    dataset = []
    traj_lengths = []
    for demo_num, object_keypoint_trajectory in object_keypoints.items():
        allegrofranka_path = args.preprocessed_data_dir + f"/demonstration_{demo_num}"

        if f"demonstration_{demo_num}" not in os.listdir(args.preprocessed_data_dir) or not os.path.exists(allegrofranka_path + "/image_indices_cam_0.pkl"):
            print(f"Skipping {allegrofranka_path} because it does not exist or doesn't have the necessary files")
            missing_files += 1
            continue

        if args.framewise_cotracker:
            H_O_C_trajectory = get_object_keypoints_in_camera_frame_framewisecotracker(trajectory=object_keypoint_trajectory, use_pixel_keypoints=dimensions==2)
        else:
            H_O_C_trajectory = get_object_keypoints_in_camera_frame_videowisecotracker(trajectory=object_keypoint_trajectory, preprocessed_data_dir=allegrofranka_path, use_pixel_keypoints=dimensions==2)
        allegrofranka,commanded_allegrofranka = get_frankallegro_in_camera_frame(preprocessed_data_dir=allegrofranka_path)
        assert len(H_O_C_trajectory) == len(allegrofranka)
        traj_lengths.append(len(allegrofranka))

        if len(allegrofranka) < args.min_length:
            print(f"Skipping {allegrofranka_path} because it has less than {args.min_length} frames")
            too_short += 1
            continue
        demonstration = {'object_keypoints': H_O_C_trajectory, 'states': allegrofranka, 'human_commanded':commanded_allegrofranka,'demo_num': demo_num}
        dataset.append(demonstration)

    for demo in dataset:
        if args.delta_actions:
            demo['actions'] = [demo['human_commanded'][i+1] - demo['human_commanded'][i] for i in range(len(demo['human_commanded'])-1)]
        else:
            demo['actions'] = [demo['human_commanded'][i+1] for i in range(len(demo['human_commanded'])-1)]

    dataset = sorted(dataset, key=lambda x: x['demo_num'])

    print(f"Generated dataset contains {len(dataset)} demonstrations")
    print(f"This excludes... Number of demonstrations skipped due to missing files: {missing_files}")
    print(f"This excludes... Number of demonstrations skipped due to being too short: {too_short}")
    print(f"Shape of the object_keypoints: {dataset[0]['object_keypoints'][0].shape}")
    print(f"Shape of the allegrofranka joint angles: {dataset[0]['states'][0].shape}")
    print(f"Shape of the actions: {dataset[0]['actions'][0].shape}")
    print("Average length of object keypoint demonstrations: ", np.mean([len(demo['object_keypoints']) for demo in dataset]))
    print("Average length of allegrofranka joint demonstrations: ", np.mean([len(demo['states']) for demo in dataset]))
    print("Average length of action sequence: ", np.mean([len(demo['actions']) for demo in dataset]))
    effective_hz = 30 // args.subsample
    dataset_name = f'{args.task_name}_{str(args.keypoints_type)}d_{args.action_type}_actions_minlength{args.min_length}_{effective_hz}hz_closed_loop_dataset.pkl'
    dataset_path = os.path.join(args.preprocessed_data_dir, dataset_name)
    with open(dataset_path, "wb") as file:
        pickle.dump(dataset, file)
    print(f"Dataset saved to {dataset_path}")

    import matplotlib.pyplot as plt
    if not os.path.exists("dataset_plots"):
        os.makedirs("dataset_plots")

    plt.hist(traj_lengths, bins=20)
    plt.xlabel("Trajectory Length")
    plt.ylabel("Frequency")
    plt.title("Histogram of Trajectory Lengths")
    plt.savefig(f"dataset_plots/histogram_of_trajectory_lengths.png")

    if dimensions == 3:
        colors = ['r', 'g', 'b', 'c']
        for demo_num in range(len(dataset)):
            #fingertips_3d = np.array(dataset[demo_num]['hand_keypoints'])
            object_keypoints_3d = dataset[demo_num]['object_keypoints']
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            '''
            for i in range(4):
                ax.plot(fingertips_3d[:, i, 0], fingertips_3d[:, i, 1], fingertips_3d[:, i, 2], label=f'Fingertip {i}', color=colors[i])
            ax.scatter(fingertips_3d[0, :, 0], fingertips_3d[0, :, 1], fingertips_3d[0, :, 2], color='green', marker='o', s=100, label='Start')
            ax.scatter(fingertips_3d[-1, :, 0], fingertips_3d[-1, :, 1], fingertips_3d[-1, :, 2], color='red', marker='*', s=100, label='Finish')
            '''
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
    elif dimensions == 2:
            print("dimensions==2 plotting code is outdated! need to write it again!")
    else:
        raise ValueError("Dimensions must be either 2 or 3")

    # Calculate the effective sampling rate in Hz
    effective_hz = 30 // args.subsample  # Assuming the default is 30 Hz

    # Adjust the dataset name to include the Hz information
    closed_loop_dataset_name = f'{args.task_name}_{str(args.keypoints_type)}d_{args.action_type}_actions_minlength{args.min_length}_{effective_hz}hz_closed_loop_dataset.pkl'

    to_return = {}
    actions = []
    all_graphs = []
    demo_nums = []
    for episode_idx in range(10):
        for start_idx in range(args.subsample):
            # Subsample actions and observations
            episode_actions = dataset[episode_idx]['actions'][start_idx::args.subsample]
            actions.append(episode_actions)
            demo_nums.append(dataset[episode_idx]['demo_num'])

            graphs = []
            for idx in range(start_idx, len(dataset[episode_idx]['actions']) + 1, args.subsample):
                concatenated = np.concatenate([
                    dataset[episode_idx]['object_keypoints'][idx].reshape(-1),
                    dataset[episode_idx]['states'][idx].reshape(-1)
                ])
                graphs.append(concatenated)
            all_graphs.append({'graph': graphs})

    to_return['actions'] = actions  # list of episodes (which themselves are lists of actions)
    to_return['observations'] = all_graphs  # list of episodes (which themselves are lists of observations, object and hand keypoints interleaved)
    to_return['demo_num'] = demo_nums
    to_return['subsample'] = args.subsample
    print("There are {} episodes in the subsampled dataset".format(len(to_return['actions'])))

    # Save the subsampled dataset
    os.makedirs("../expert_demos/general", exist_ok=True)
    os.makedirs("../processed_data/points", exist_ok=True)
    output_path_expert = f"../expert_demos/general/{closed_loop_dataset_name}"
    output_path_processed = f"../processed_data/points/{closed_loop_dataset_name}"

    with open(output_path_expert, "wb") as f_expert, open(output_path_processed, "wb") as f_processed:
        pickle.dump(to_return, f_expert)
        pickle.dump(to_return, f_processed)

    print(f"Dumped closed loop dataset for {closed_loop_dataset_name} to expert_demos and processed_data folders")
