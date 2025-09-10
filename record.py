import argparse
import logging
from pynput import keyboard
import cv2
import numpy as np

from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig


def init_listeners():
	event = {
		"repeat": False,
		"stop": False,
		"next": False,
	}
	def on_press(key):
		if key == keyboard.Key.delete:
			event["repeat"] = True
		if key == keyboard.Key.esc:
			event["stop"] = True
		if key == keyboard.Key.enter:
			event["next"] = True
	listener = keyboard.Listener(on_press=on_press)
	listener.start()
	return listener, event

def main():
	parser = argparse.ArgumentParser()
       
	parser.add_argument(
		"--episode-index",
		type=int,
		required=True,
        )
	parser.add_argument(
                "--total-episodes-number",
                type=int,
                required=True,
        )
	args = parser.parse_args()
	episode_ind = args.episode_index
	total_episodes_num = args.total_episodes_number
	
	camera_config = {
	    "third_person_view": OpenCVCameraConfig(index_or_path=0, width=1280, height=720, fps=30)
	}

	boss_config = SO101LeaderConfig(
	    port="/dev/tty.usbmodem58FA0919711",
	    id="koval_los",
	)


	rab_config = SO101FollowerConfig(
	    port="/dev/tty.usbmodem58FA0930111",
	    id="koval_pes",
	    cameras=camera_config
	)


	rab = SO101Follower(rab_config)
	boss = SO101Leader(boss_config)





	listener, event = init_listeners()
	listener.stop()


def record_episode(event):
	while True:	
		if event["stop"] or event["delete"]:
			break
		if event["next"]:
			# save data
			break
			
	logging.info(f"Last demo to save: #{demo_number - 1}")
	listener.stop()



if __name__ == "__main__":
    main()
