import argparse
import logging
import os
import time

from pynput import keyboard
import cv2
import numpy as np
import h5py

from system_config import get_leader, get_follower, CONFIG


def wait_(seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def init_listeners():
    event = {
        "repeat": False,
        "stop": False,
        "next": False,
    }
    def on_press(key):
        if key == keyboard.Key.backspace:
            event["repeat"] = True
        if key == keyboard.Key.esc:
            event["stop"] = True
        if key == keyboard.Key.enter:
            event["next"] = True
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, event


def record_episode(
        config,
        event
):
    episode_data = list()
    while True:
        loop_start = time.perf_counter()
        if event["stop"] or event["repeat"] or event["next"]:
            return episode_data, event
        leader_obs = config['leader'].get_action()
        follower_obs = config['follower'].get_observation()
        data = {
            'robot_state': [leader_obs['shoulder_pan.pos'], leader_obs['shoulder_lift.pos'], leader_obs['elbow_flex.pos'],
             leader_obs['wrist_flex.pos'], leader_obs['wrist_roll.pos'], leader_obs['gripper.pos']],
        }
        for cam_key, _ in config['follower'].cameras.items():
            data[cam_key] = follower_obs[cam_key]
        episode_data.append(data)
        loop_time = time.perf_counter() - loop_start
        wait_(1 / config['fps'] - loop_time)


def save_episode_data(episode_data, episode_num, config):
    logging.info(f"Saving recording for {episode_num}")
    os.makedirs(config['save_path'], exist_ok=True)
    with h5py.File(os.path.join(config['save_path'], f"{episode_num}.hdf5"), "w") as f:
        robot_state = np.array([x['robot_state'] for x in episode_data], dtype=np.float32)
        f.create_dataset("robot_state", data=robot_state)

        for cam_key, _ in config['follower'].cameras.items():
            camera_data = np.array([x[cam_key] for x in episode_data], dtype=np.float32)
            f.create_dataset(cam_key, data=camera_data)

    logging.info(f"Recording for {episode_num} is saved")


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
       
    parser.add_argument(
        "--episode-number",
        type=int,
        required=True,
        )
    parser.add_argument(
                "--total-episodes-number",
                type=int,
                required=True,
        )
    parser.add_argument(
                "--fps",
                type=int,
                required=True,
        )
    parser.add_argument(
                "--save-path",
                type=str,
                required=True,
        )
    args = parser.parse_args()
    episode_num = args.episode_number
    total_episodes_num = args.total_episodes_number

    rab = get_follower()
    rab.connect()
    boss = get_leader()
    boss.connect()
    listener, event = init_listeners()

    record_config = {
        'leader': boss,
        'follower': rab,
        'fps': args.fps,
        'save_path': args.save_path,
    }
    logging.info(f"System is ready. Recording {total_episodes_num} episodes")

    while episode_num <= total_episodes_num:
        episode_data, status = record_episode(record_config, event)
        if status['stop']:
            logging.info(f"Stopped recording")
            break
        if status['repeat']:
            logging.info(f"Repeating recording for {episode_num}")
            event['repeat'] = False
            continue

        event['next'] = False
        save_episode_data(episode_data, episode_num, record_config)
        episode_num += 1


    listener.stop()
    rab.disconnect()
    boss.disconnect()






if __name__ == "__main__":
    main()
