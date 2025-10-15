import zmq
import json
import struct
import cv2
import numpy as np
import logging
import argparse

import torch
from sklearn.model_selection import train_test_split
from transformers import AutoProcessor, AutoModel, SiglipVisionModel
from torch.utils.data import DataLoader, random_split

from policy.config import POLICY_CONFIG
from policy.dataset import CustomDataset
from policy.model import Trener

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights-path",
        type=str,
        required=True,
        )
    args = parser.parse_args()
    weights_path = args.weights_path


    context = zmq.Context()
    socket_in = context.socket(zmq.PULL)
    socket_in.bind("tcp://*:5555")
    socket_out = context.socket(zmq.PUSH)
    socket_out.connect("tcp://10.40.33.115:5554")
    logging.info("Inference Server started.")

    """
    model_name = 'google/siglip-so400m-patch14-224'
    vision_encoder = SiglipVisionModel.from_pretrained(model_name)
    img_processor = AutoProcessor.from_pretrained(model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'#### ------> Using device: {device}')
    policy = Trener(vision_encoder, POLICY_CONFIG)
    policy.to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    """

    while True:
        parts = socket_in.recv_multipart()
        logging.info("Message received")

        input_data = dict()
        input_data['robot_state'] = torch.tensor(struct.unpack("6f", parts[0]))

        meta = json.loads(parts[1].decode("utf-8"))
        camera_meta = meta["images"]
        for i, info in enumerate(camera_meta):
            cam_key = info["key"]
            jpg_bytes = parts[2 + i]
            img = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
            #proc = img_processor(images=img, return_tensors="pt")["pixel_values"].squeeze()
            input_data[cam_key] = img
            #cv2.imshow(cam_key, img)
            #cv2.waitKey(1)

        prediction  = [.228] * POLICY_CONFIG['actions_dim'] * POLICY_CONFIG['prediction_horizon']#= policy(batch)
        msg_len = len(prediction)
        msg = struct.pack(f"{msg_len}f", *prediction)
        socket_out.send_multipart([
            struct.pack("i", msg_len),
            msg
        ])


if __name__ == "__main__":
    main()
