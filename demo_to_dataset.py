import argparse
import logging
import os
import csv

import cv2
import numpy as np
import h5py


def save_jpeg(jpeg_bytes, filename, save_dir):
    with open(os.path.join(save_dir, filename + '.jpg'), 'wb') as f:
        f.write(jpeg_bytes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demos-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        required=True,
    )
    args = parser.parse_args()
    demos_path = args.demos_path
    save_path = os.path.join(demos_path, 'dataset')
    chunk_size = args.chunk_size

    os.makedirs(save_path, exist_ok=True)
    cnt = 0

    with open(os.path.join(save_path, "states.csv"), "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'robot_state', 'actions'])

    for filename in os.listdir(demos_path):
        print(filename)

        if '.hdf5' not in filename:
            continue
        with h5py.File(os.path.join(demos_path, filename), "r") as f:
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
                chunk.append(' '.join(map(str, action)))
            robot_states_cvs.append([ind + cnt,  ' '.join(map(str,state)), ' '.join(chunk)])
        print(len(robot_states_cvs))
        with open(os.path.join(save_path, "states.csv"), "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(robot_states_cvs)

        cnt += dp_num

if __name__ == "__main__":
    main()
