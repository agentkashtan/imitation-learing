import logging
import os
import argparse

import torch
from sklearn.model_selection import train_test_split
from transformers import AutoProcessor, AutoModel, SiglipVisionModel
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from policy.dataset import CustomDataset
from policy.model import Trener
from system_config import CONFIG


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint-filename",
        type=str,
        required=False,
    )
    args = parser.parse_args()
    checkpoint_filename = args.checkpoint_filename

    WEIGHTS_PATH = './weights'
    logging.basicConfig(level=logging.INFO)
    batch_size = CONFIG['training_config'].batch_size
    os.makedirs(WEIGHTS_PATH, exist_ok=True)
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
    policy = Trener(vision_encoder, CONFIG['training_config'])
    policy.to(device)
    # continue from checkpoint
    if checkpoint_filename is not None:
        checkpoint = torch.load(os.path.join(WEIGHTS_PATH, checkpoint_filename), map_location=device)
        policy.load_state_dict(checkpoint["model_state_dict"])
        epoch_num = checkpoint["epoch_num"]
    else:
        epoch_num = 1
    
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=CONFIG['training_config'].lr,
        betas=(CONFIG['training_config'].beta1, CONFIG['training_config'].beta2),
        eps=CONFIG['training_config'].eps
    )
    loss_fn = torch.nn.MSELoss()
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logging.info(f'### ------> Total params: {total_params}; trainable params: {trainable_params}')
    logging.info(f'#### ------> Starting training: {CONFIG["training_config"].epoch_num} epochs')
    for _ in range(CONFIG['training_config'].epoch_num):
        policy.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'processing epoch: {epoch_num}', leave=False):
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
        print(f"Epoch {epoch_num:02d}/{CONFIG['training_config'].epoch_num} "
              f"| Train Loss: {avg_train_loss:.4f} "
              f"| Val Loss: {avg_val_loss:.4f}")
        if epoch_num % 1 == 0 or epoch_num == CONFIG['training_config'].epoch_num:
            torch.save({
                "epoch": epoch_num,
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_loss": total_loss / max(1, len(train_loader)),
                "validation_loss": val_total_loss / max(1, len(val_loader))
            }, f'{WEIGHTS_PATH}/model_weights_epoch_{epoch_num}.pt')
        epoch_num += 1


if __name__ == "__main__":
    train()
