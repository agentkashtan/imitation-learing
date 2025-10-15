import argparse
import logging
import os
import time

import cv2
import numpy as np
import h5py

from utils import get_follower, robot_state_ind_to_names, decode_jpeg, wait_
from system_config import CONFIG


def load_episode_data(save_path, episode_num):
    with h5py.File(os.path.join(save_path, f"{episode_num}.hdf5"), "r") as f:
        robot_states = np.array(f['robot_state'], dtype=np.float32)
        cameras_data = dict()
        for cam_key in list(f.keys()):
            if cam_key == 'robot_state_follower' or cam_key == 'robot_state_leader':
                continue
            frames = [decode_jpeg(jpeg_bytes) for jpeg_bytes in f[cam_key]]
            cameras_data[cam_key] = np.stack(frames, axis=0)

    return robot_states, cameras_data

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episode-number",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    rab = get_follower()
    rab.connect()
    robot_states, cameras_data = load_episode_data(args.save_path, args.episode_number)
    for ind, robot_state in enumerate(robot_states):
        rab.send_action(robot_state_ind_to_names(robot_state))
        for cam_key, obs in cameras_data.items():
            img = cv2.cvtColor(obs[ind], cv2.COLOR_BGR2RGB)
            cv2.imshow(cam_key, img)
        cv2.waitKey(1000 // CONFIG['fps'])

    rab.disconnect()

if __name__ == "__main__":
    main()
