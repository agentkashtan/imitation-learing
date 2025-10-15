import logging
import os

import torch
from sklearn.model_selection import train_test_split
from transformers import AutoProcessor, AutoModel, SiglipVisionModel
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from policy.config import POLICY_CONFIG
from policy.dataset import CustomDataset
from policy.model import Trener


def train():
    os.makedirs('./weights', exist_ok=True)
    model_name = 'google/siglip-so400m-patch14-224'
    vision_encoder = SiglipVisionModel.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)

    dataset = CustomDataset(
        './demos/demos/dataset/states.csv',
        './demos/demos/dataset',
        ['third_person_view', 'wrist_view'],
        processor
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    policy = Trener(vision_encoder, POLICY_CONFIG)
    policy.to(device)

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=POLICY_CONFIG["lr"],
        betas=(POLICY_CONFIG["beta1"], POLICY_CONFIG["beta2"]),
        eps=POLICY_CONFIG["eps"]
    )
    loss_fn = torch.nn.MSELoss()
    for epoch in range(POLICY_CONFIG["epoch_num"]):
        policy.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'processing epoch: {epoch + 1}'):
            optimizer.zero_grad()
            del batch['index']
            for key, value in batch.items():
                batch[key] = value.to(device)
            prediction = policy(batch)
            prediction_flat = prediction.view(prediction.shape[0], -1)
            loss = loss_fn(prediction_flat, batch['actions'])
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print("Training loss:", total_loss / len(train_loader))

        policy.eval()
        val_total_loss = 0
        for batch in tqdm(val_loader):
            for key, value in batch.items():
                batch[key] = value.to(device)
            with torch.no_grad():
                prediction = policy(batch)
                prediction_flat = prediction.view(prediction.shape[0], -1)
                loss = loss_fn(prediction_flat, batch['actions'])
                val_total_loss += loss.item()
        print("Validation loss:", val_total_loss / len(val_loader))

        if (epoch + 1) % 5 == 0 or epoch + 1 == num_epoch:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_loss": total_loss / max(1, len(train_loader)),
                "validation_loss": val_total_loss / max(1, len(val_loader))
            }, f'{WEIGHTS_PATH}/model_weights_epoch_{epoch + 1}.pt')


if __name__ == "__main__":
    train()
