import argparse
import logging
import os
import csv

import cv2
import numpy as np
import h5py

from system_config import CONFIG


def save_jpeg(jpeg_bytes, filename, save_dir):
    with open(os.path.join(save_dir, filename + '.jpg'), 'wb') as f:
        f.write(jpeg_bytes)


def compute_stats(demos_path, chunk_size):
    all_states = []
    for filename in os.listdir(demos_path):
        if not filename.endswith(".hdf5"):
            continue

        file_path = os.path.join(demos_path, filename)
        with h5py.File(file_path, "r") as f:
            #TODO fix robot_state field
            #robot_states = np.array(f[CONFIG['robot_state_field']], dtype=np.float32)
            robot_states = np.array(f['robot_state'], dtype=np.float32)
            dp_num = len(robot_states) - chunk_size
            all_states.append(robot_states[:dp_num])

    all_states = np.vstack(all_states)

    mean = np.mean(all_states, axis=0)
    std = np.std(all_states, axis=0)

    print("Mean per joint:", mean)
    print("Std per joint:", std)
    return mean, std

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demos-path",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    demos_path = args.demos_path
    save_path = os.path.join(demos_path, 'dataset')
    chunk_size = CONFIG['training_config'].prediction_horizon
    os.makedirs(save_path, exist_ok=True)
    cnt = 0
    mean, std = compute_stats(demos_path, chunk_size)

    with open(os.path.join(save_path, "states.csv"), "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'robot_state', 'actions'])

    for filename in os.listdir(demos_path):

        if '.hdf5' not in filename:
            continue
        with h5py.File(os.path.join(demos_path, filename), "r") as f:
            # TODO fix robot_state field
            #robot_states = np.array(f[CONFIG['robot_state_field']], dtype=np.float32)
            robot_states = np.array(f['robot_state'], dtype=np.float32)
            dp_num = len(robot_states) - chunk_size
            for cam_key in list(f.keys()):
                if cam_key == 'robot_state':
                    continue
                local_cnt = 0
                for frame in f[cam_key][:dp_num]:
                    save_jpeg(frame, f'{cam_key}_{cnt + local_cnt}', save_path)
                    local_cnt += 1

        robot_states_cvs = list()
        for ind, state in enumerate(robot_states[:dp_num]):
            chunk = list()
            for action in robot_states[ind:ind + chunk_size]:
                normalized_action = [(val - mean[j]) / std[j] for j,val in enumerate(action)]
                chunk.append(' '.join(map(str, normalized_action)))

            normalized_state = [(val - mean[j]) / std[j] for j,val in enumerate(state)]
            robot_states_cvs.append([ind + cnt,  ' '.join(map(str,normalized_state)), ' '.join(chunk)])

        with open(os.path.join(save_path, "states.csv"), "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(robot_states_cvs)

        cnt += dp_num

if __name__ == "__main__":
    main()
