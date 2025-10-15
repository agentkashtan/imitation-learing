import zmq
import cv2
import numpy as np
import struct
import logging
import json

from system_config import get_leader, get_follower, CONFIG, robot_state_names_to_ind, robot_state_ind_to_names, wait_


RUN_CONFIG = {
    'fps': 50,
    'mean': [ 14.548566, -51.984997 , 57.5864  ,  59.111923,   2.572637 , 18.008852],
    'std': [27.007866 ,35.0966 ,  23.20677,  10.410391 , 7.636047 ,10.38727 ],
}

def normalize_action(action, mean, std):
    return [(val - mean[j]) / std[j] for j,val in enumerate(action)]

def denormalize_action(action, mean, std):
    return [val  * std[j] + mean[j] for j, val in enumerate(action)]


def run_actions(actions, follower):
    for action in actions:
        follower.send_action(
            robot_state_ind_to_names(
                denormalize_action(
                    action,
                    RUN_CONFIG['mean'],
                    RUN_CONFIG['std']
                )
            )
        )
        wait_(1 / RUN_CONFIG['fps'])

def get_next_actions(observation, socket_in, socket_out):
    observation["robot_state"] = normalize_action(observation["robot_state"], RUN_CONFIG['mean'], RUN_CONFIG['std'])
    robot_state_bytes = struct.pack("6f", *observation["robot_state"])

    camera_meta = []
    image_frames = []

    for cam_key, img in observation["images"].items():
        _, buf = cv2.imencode(".jpg", img)
        jpg_bytes = buf.tobytes()

        camera_meta.append({
            "key": cam_key,
            "size": len(jpg_bytes)
        })

        image_frames.append(jpg_bytes)

    meta_bytes = json.dumps({"images": camera_meta}).encode("utf-8")

    socket_out.send_multipart([robot_state_bytes, meta_bytes, *image_frames])
    logging.info('Message sent')

    parts = socket_in.recv_multipart()
    num_floats = struct.unpack("i", parts[0])[0]
    actions_flatten = struct.unpack(f"{num_floats}f", parts[1])
    actions = np.array(actions_flatten, dtype=np.float32).reshape(num_floats // 6, 6)

    return actions


def main():
    logging.basicConfig(level=logging.INFO)
    context = zmq.Context()
    socket_out = context.socket(zmq.PUSH)
    socket_out.connect("tcp://127.0.0.1:5555")
    socket_in = context.socket(zmq.PULL)
    socket_in.connect("tcp://127.0.0.1:5554")

    logging.info(f"Connected to the inference server")

    follower = get_follower()
    follower.connect()
    logging.info(f"Connected to the robot")

    while True:
        observation = follower.get_observation()
        current_state = {
            'robot_state': robot_state_names_to_ind(observation),
            'images': {}
        }
        for cam_key, _ in follower.cameras.items():
            current_state['images'][cam_key] = observation[cam_key]

        actions = get_next_actions(current_state, socket_in, socket_out)
        run_actions(actions, follower)

    follower.disconnect()

if __name__ == "__main__":
    main()