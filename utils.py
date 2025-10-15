import cv2
import numpy as np
import time

from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from camera import Camera
from robot import Follower
from system_config import CONFIG


def robot_state_names_to_ind(robot_state):
    return [
        robot_state['shoulder_pan.pos'],
        robot_state['shoulder_lift.pos'],
        robot_state['elbow_flex.pos'],
        robot_state['wrist_flex.pos'],
        robot_state['wrist_roll.pos'],
        robot_state['gripper.pos']
    ]


def robot_state_ind_to_names(robot_state):
    return {
        'shoulder_pan.pos': robot_state[0],
        'shoulder_lift.pos': robot_state[1],
        'elbow_flex.pos': robot_state[2],
        'wrist_flex.pos': robot_state[3],
        'wrist_roll.pos': robot_state[4],
        'gripper.pos': robot_state[5]
    }


def get_follower():
    cameras = dict()
    for cam_key, cam_config in CONFIG['hardware_config']['cameras'].items():
        print(cam_config)
        cameras[cam_key] = Camera(cam_config)
    follower_config = SO101FollowerConfig(**CONFIG['hardware_config']['follower_config'])
    return Follower(SO101Follower(follower_config), cameras)


def get_leader():
    leader_config = SO101LeaderConfig(**CONFIG['hardware_config']['leader_config'])
    return SO101Leader(leader_config)


def decode_jpeg(jpeg_bytes):
    return cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR)


def wait_(seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def normalize_action(action, mean, std):
    return [(val - mean[j]) / std[j] for j,val in enumerate(action)]


def denormalize_action(action, mean, std):
    return [val  * std[j] + mean[j] for j, val in enumerate(action)]