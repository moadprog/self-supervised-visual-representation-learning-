import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datasets import get_stl10_pretrain_dataset
from src.models.simclr import SimCLR
from src.losses.nt_xent import NTXentLoss
from torch.utils.tensorboard import SummaryWriter
import subprocess

# --------------------------------------------------
# Configuration
# --------------------------------------------------

batch_size = 256
epochs = 1
learning_rate = 3e-4
temperature = 0.5
num_workers = 4

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

writer = SummaryWriter(log_dir="runs/stl10_simclr")
# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = get_stl10_pretrain_dataset(
    root="data",
    image_size=96,
)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True,
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = SimCLR(
    projection_dim=128,
    projection_hidden_dim=2048,
).to(device)


# --------------------------------------------------
# Loss
# --------------------------------------------------

criterion = NTXentLoss(
    temperature=temperature
)


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-4,
)


# --------------------------------------------------
# Mixed precision
# --------------------------------------------------

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(device.type == "cuda"),
)


# --------------------------------------------------
# Checkpoints
# --------------------------------------------------

checkpoint_dir = "checkpoints/stl10"
os.makedirs(checkpoint_dir, exist_ok=True)


# --------------------------------------------------
# Training
# --------------------------------------------------

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    global_step = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1}/{epochs}",
    )

    for views, _ in progress_bar:

        x1, x2 = views

        x1 = x1.to(
            device,
            non_blocking=True,
        )

        x2 = x2.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad()

        with torch.amp.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda"),
        ):

            _, z1 = model(x1)
            _, z2 = model(x2)

            loss = criterion(z1, z2)

            writer.add_scalar(
                            "Loss/train_step",
                            loss.item(),
                            global_step,
            )

            global_step += 1

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    average_loss = running_loss / len(dataloader)

    writer.add_scalar(
                    "Loss/train_epoch",
                    average_loss,
                    epoch + 1,
    )

    print(
        f"Epoch {epoch + 1}: "
        f"average loss = {average_loss:.4f}"
    )

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"epoch_{epoch + 1:03d}.pt",
    )

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "loss": average_loss,
        },
        checkpoint_path,
    )

    s3_path = (
            "s3://lachqar/"
            "self-supervised-visual-representation-learning/"
            "checkpoints/stl10/"
            f"epoch_{epoch + 1:03d}.pt"
    )

    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            checkpoint_path,
            s3_path,
            "--endpoint-url",
            os.environ["AWS_ENDPOINT_URL"],
        ],
        check=True,
    )

writer.close()
