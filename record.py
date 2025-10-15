import argparse
import logging
import os
import time

from pynput import keyboard
import cv2
import numpy as np
import h5py

from utils import get_leader, get_follower, robot_state_names_to_ind, wait_
from system_config import CONFIG

def init_listeners():
    event = {
        "repeat": False,
        "stop": False,
        "next": False,
        "start": False,
    }
    def on_press(key):
        event["repeat"] = False
        event["stop"] = False
        event["next"] = False
        event["start"] = False

        if key == keyboard.Key.backspace:
            event["repeat"] = True
        if key == keyboard.Key.esc:
            event["stop"] = True
        if key == keyboard.Key.enter:
            event["next"] = True
        if key == keyboard.Key.space:
            event["start"] = True

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, event


def record_episode(
        config,
        event
):
    episode_data = list()
    window_positions = {
        cam_key: {
            'value': False,
            'offset': i * 700
        } for i, (cam_key, _)  in enumerate(config['follower'].cameras.items())
    }
    while True:
        loop_start = time.perf_counter()
        if event["stop"] or event["repeat"] or event["next"]:
            return episode_data, event
        follower_obs = config['follower'].get_observation()
        action = config['leader'].get_action()
        config['follower'].send_action(action)
        data = {
            'robot_state_follower': robot_state_names_to_ind(follower_obs),
            'robot_state_leader': robot_state_names_to_ind(action),
        }
        for cam_key, _ in config['follower'].cameras.items():
            data[cam_key] = follower_obs[cam_key]
            img = data[cam_key]
            cv2.imshow(cam_key, img)
            if not window_positions[cam_key]['value']:
                window_positions[cam_key]['value'] = True
                cv2.moveWindow(cam_key, window_positions[cam_key]['offset'], 100)
        cv2.waitKey(1)
        episode_data.append(data)
        loop_time = time.perf_counter() - loop_start
        wait_(1 / config['fps'] - loop_time)

def save_episode_data(episode_data, episode_num, config):
    logging.info(f"Saving recording for {episode_num}")
    os.makedirs(config['save_path'], exist_ok=True)
    with h5py.File(os.path.join(config['save_path'], f"{episode_num}.hdf5"), "w") as f:
        robot_state = np.array([x['robot_state'] for x in episode_data], dtype=np.float32)
        f.create_dataset("robot_state", data=robot_state)
        logging.info(f"Episode #{episode_num}; saving {len(robot_state)} data points")
        for cam_key, _ in config['follower'].cameras.items():
            encoded_frames = []
            for data in episode_data:
                success, encoded_img = cv2.imencode(".jpg", data[cam_key])
                if not success:
                    logging.error("Failed to encode frame")
                encoded_frames.append(encoded_img.flatten())
            dt = h5py.vlen_dtype(np.dtype("uint8"))
            f.create_dataset(cam_key, data=encoded_frames, dtype=dt)

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
        'fps': CONFIG['fps'],
        'save_path': args.save_path,
    }
    logging.info(f"\n\nSystem is ready. Recording {total_episodes_num} episodes")

    while episode_num <= total_episodes_num:
        if event['start']:
            event['start'] = False
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
