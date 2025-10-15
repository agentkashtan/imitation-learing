from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import cv2


class CustomDataset(Dataset):
    def __init__(self, rs_dataset_path, imgs_path, cam_keys, img_processor):
        super().__init__()
        self.imgs_path = imgs_path
        self.cam_keys = cam_keys
        self.img_processor = img_processor

        df = pd.read_csv(rs_dataset_path)
        self.dataset = []
        for _, row in df.iterrows():
            self.dataset.append({
                "index": int(row["index"]),
                "robot_state": torch.tensor(
                    np.array(str(row["robot_state"]).split(), dtype=np.float32)
                ),
                "actions": torch.tensor(
                    np.array(str(row["actions"]).split(), dtype=np.float32)
                ),
            })

    def __getitem__(self, index):
        item = dict(self.dataset[index])
        for cam_key in self.cam_keys:
            img = cv2.imread(f"{self.imgs_path}/{cam_key}_{item['index']}.jpg")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            proc = self.img_processor(images=img, return_tensors="pt")["pixel_values"].squeeze()
            item[cam_key] = proc
        return item

    def __len__(self):
        return len(self.dataset)
