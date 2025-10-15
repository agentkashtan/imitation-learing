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

from system_config import CONFIG
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
    socket_out.bind("tcp://*:5554")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = 'google/siglip-so400m-patch14-224'
    vision_encoder = SiglipVisionModel.from_pretrained(model_name)
    img_processor = AutoProcessor.from_pretrained(model_name)

    logging.info(f'####------> Using device: {device}')
    policy = Trener(vision_encoder, CONFIG['training_config'])
    policy.to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    logging.info("Inference Server started.")

    while True:
        parts = socket_in.recv_multipart()
        logging.info("Message received")

        input_data = dict()
        input_data['robot_state'] = torch.tensor(struct.unpack("6f", parts[0])).unsqueeze(0)

        meta = json.loads(parts[1].decode("utf-8"))
        camera_meta = meta["images"]
        for i, info in enumerate(camera_meta):
            cam_key = info["key"]
            jpg_bytes = parts[2 + i]
            img = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            proc = img_processor(images=img, return_tensors="pt")["pixel_values"]
            input_data[cam_key] = proc

        for key, value in input_data.items():
            input_data[key] = value.to(device)
        with torch.no_grad():
            prediction = policy(input_data)
            prediction = prediction.view(prediction.shape[0], -1).squeeze().cpu().numpy()
        msg_len = prediction.size
        msg = struct.pack(f"{msg_len}f", *prediction)
        socket_out.send_multipart([
            struct.pack("i", msg_len),
            msg
        ])

if __name__ == "__main__":
    main()
