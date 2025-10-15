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
    logging.basicConfig(level=logging.INFO)
    batch_size = POLICY_CONFIG["batch_size"]
    os.makedirs('./weights', exist_ok=True)
    WEIGHTS_PATH = './weights'
    model_name = 'google/siglip-so400m-patch14-224'
    vision_encoder = SiglipVisionModel.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    logging.info(f"#### ------> Loaded vision encoder")
    dataset = CustomDataset(
        './demos/demos/dataset/states.csv',
        './demos/demos/dataset',
        ['third_person_view', 'wrist_view'],
        processor
    )
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    logging.info(f'#### ------> Loaded dataset: training size: {train_dataset.__len__()}; batch size: {batch_size}; number of batches: {len(train_loader)}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'#### ------> Using device: {device}')
    policy = Trener(vision_encoder, POLICY_CONFIG)
    policy.to(device)
    # continue from checkpoint
    checkpoint = torch.load(f"{WEIGHTS_PATH}/model_weights_epoch_40.pt", map_location=device)

    # Restore model weights
    policy.load_state_dict(checkpoint["model_state_dict"])
    
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=POLICY_CONFIG["lr"],
        betas=(POLICY_CONFIG["beta1"], POLICY_CONFIG["beta2"]),
        eps=POLICY_CONFIG["eps"]
    )
    loss_fn = torch.nn.MSELoss()
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logging.info(f'### ------> Total params: {total_params}; trainable params: {trainable_params}')
    logging.info(f'#### ------> Starting training: {POLICY_CONFIG["epoch_num"]} epochs')
    for epoch in range(POLICY_CONFIG["epoch_num"]):
        policy.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'processing epoch: {epoch + 1}', leave=False):
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

        policy.eval()
        val_total_loss = 0
        for batch in tqdm(val_loader, leave=False):
            for key, value in batch.items():
                batch[key] = value.to(device)
            with torch.no_grad():
                prediction = policy(batch)
                prediction_flat = prediction.view(prediction.shape[0], -1)
                loss = loss_fn(prediction_flat, batch['actions'])
                val_total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_total_loss / len(val_loader)
        print(f"Epoch {epoch+1:02d}/{POLICY_CONFIG['epoch_num']} "
              f"| Train Loss: {avg_train_loss:.4f} "
              f"| Val Loss: {avg_val_loss:.4f}")
        if (epoch + 1) % 1 == 0 or epoch + 1 == POLICY_CONFIG["epoch_num"]:
            torch.save({
                "epoch": epoch + 1 + 40,
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_loss": total_loss / max(1, len(train_loader)),
                "validation_loss": val_total_loss / max(1, len(val_loader))
            }, f'{WEIGHTS_PATH}/model_weights_epoch_{epoch + 1 + 40}.pt')


if __name__ == "__main__":
    train()
