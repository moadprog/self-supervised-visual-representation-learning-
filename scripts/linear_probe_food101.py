import os
import subprocess

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import Food101
from tqdm import tqdm

from src.models.encoder import ResNet50Encoder
from src.models.simclr import SimCLR


# ============================================================
# Configuration
# ============================================================

feature_batch_size = 128
linear_batch_size = 512

linear_epochs = 30
linear_lr = 0.1
momentum = 0.9

num_workers = 4
num_classes = 101

seed = 42


# ============================================================
# Paths
# ============================================================

data_root = "/tmp/food101"

checkpoint_dir = "/tmp/simclr_checkpoints/food101_linear_probe"
os.makedirs(checkpoint_dir, exist_ok=True)

simclr_checkpoint_path = (
    "/tmp/simclr_epoch100.pt"
)

s3_simclr_checkpoint = (
    "s3://lachqar/"
    "self-supervised-visual-representation-learning/"
    "checkpoints/"
    "food101_simclr_bs128_100ep/"
    "epoch_100.pt"
)

endpoint = os.environ["AWS_ENDPOINT_URL"]


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )


# ============================================================
# Food-101 deterministic evaluation transform
# ============================================================

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# Dataset
# ============================================================

train_dataset = Food101(
    root=data_root,
    split="train",
    transform=eval_transform,
    download=False,
)

test_dataset = Food101(
    root=data_root,
    split="test",
    transform=eval_transform,
    download=False,
)

print(
    "Train images:",
    len(train_dataset),
)

print(
    "Test images:",
    len(test_dataset),
)


# ============================================================
# DataLoaders used to extract frozen representations
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=feature_batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=feature_batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)


# ============================================================
# Download trained SimCLR checkpoint from S3
# ============================================================

if not os.path.exists(simclr_checkpoint_path):

    print(
        "\nDownloading epoch-100 SimCLR checkpoint..."
    )

    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            s3_simclr_checkpoint,
            simclr_checkpoint_path,
            "--endpoint-url",
            endpoint,
        ],
        check=True,
    )

    print(
        "Checkpoint downloaded.\n"
    )


# ============================================================
# Feature extraction
# ============================================================

@torch.inference_mode()
def extract_features(
    encoder,
    dataloader,
    description,
):

    encoder.eval()

    all_features = []
    all_labels = []

    progress_bar = tqdm(
        dataloader,
        desc=description,
    )

    for images, labels in progress_bar:

        images = images.to(
            device,
            non_blocking=True,
        )

        features = encoder(images)

        all_features.append(
            features.cpu()
        )

        all_labels.append(
            labels.cpu()
        )

    features = torch.cat(
        all_features,
        dim=0,
    )

    labels = torch.cat(
        all_labels,
        dim=0,
    )

    return features, labels


# ============================================================
# Linear classifier training
# ============================================================

def train_linear_classifier(
    train_features,
    train_labels,
    test_features,
    test_labels,
    experiment_name,
):

    print(
        "\n========================================"
    )
    print(
        f"Linear probe: {experiment_name}"
    )
    print(
        "========================================\n"
    )

    train_feature_dataset = TensorDataset(
        train_features,
        train_labels,
    )

    test_feature_dataset = TensorDataset(
        test_features,
        test_labels,
    )

    train_feature_loader = DataLoader(
        train_feature_dataset,
        batch_size=linear_batch_size,
        shuffle=True,
    )

    test_feature_loader = DataLoader(
        test_feature_dataset,
        batch_size=linear_batch_size,
        shuffle=False,
    )


    # --------------------------------------------------------
    # Linear classifier ONLY
    # --------------------------------------------------------

    classifier = nn.Linear(
        2048,
        num_classes,
    ).to(device)


    criterion = nn.CrossEntropyLoss()


    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=linear_lr,
        momentum=momentum,
        weight_decay=0.0,
    )


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=linear_epochs,
        eta_min=1e-4,
    )


    best_accuracy = 0.0


    # ========================================================
    # Linear probe epochs
    # ========================================================

    for epoch in range(linear_epochs):

        classifier.train()

        running_loss = 0.0
        correct = 0
        total = 0


        for features, labels in train_feature_loader:

            features = features.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            logits = classifier(
                features
            )


            loss = criterion(
                logits,
                labels,
            )


            loss.backward()

            optimizer.step()


            running_loss += (
                loss.item()
                * labels.size(0)
            )


            predictions = logits.argmax(
                dim=1
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += labels.size(0)


        scheduler.step()


        train_loss = (
            running_loss / total
        )

        train_accuracy = (
            100.0 * correct / total
        )


        # ====================================================
        # Test evaluation
        # ====================================================

        classifier.eval()

        test_correct = 0
        test_total = 0


        with torch.inference_mode():

            for features, labels in test_feature_loader:

                features = features.to(
                    device,
                    non_blocking=True,
                )

                labels = labels.to(
                    device,
                    non_blocking=True,
                )


                logits = classifier(
                    features
                )


                predictions = logits.argmax(
                    dim=1
                )


                test_correct += (
                    predictions == labels
                ).sum().item()


                test_total += labels.size(0)


        test_accuracy = (
            100.0
            * test_correct
            / test_total
        )


        best_accuracy = max(
            best_accuracy,
            test_accuracy,
        )


        print(
            f"Epoch {epoch + 1:02d}/{linear_epochs} "
            f"| loss={train_loss:.4f} "
            f"| train_acc={train_accuracy:.2f}% "
            f"| test_acc={test_accuracy:.2f}%"
        )


    print(
        "\nBest test accuracy:",
        f"{best_accuracy:.2f}%"
    )


    return best_accuracy


# ============================================================
# EXPERIMENT 1
# Random frozen ResNet-50
# ============================================================

print(
    "\n========================================"
)

print(
    "Extracting RANDOM ResNet-50 features"
)

print(
    "========================================\n"
)


random_encoder = ResNet50Encoder().to(
    device
)


for parameter in random_encoder.parameters():
    parameter.requires_grad = False


random_train_features, train_labels = (
    extract_features(
        random_encoder,
        train_loader,
        "Random encoder - train",
    )
)


random_test_features, test_labels = (
    extract_features(
        random_encoder,
        test_loader,
        "Random encoder - test",
    )
)


print(
    "Random train features:",
    random_train_features.shape,
)

print(
    "Random test features:",
    random_test_features.shape,
)


random_accuracy = train_linear_classifier(
    random_train_features,
    train_labels,
    random_test_features,
    test_labels,
    "Random frozen ResNet-50",
)


# Free GPU memory before loading SimCLR
del random_encoder

torch.cuda.empty_cache()


# ============================================================
# EXPERIMENT 2
# SimCLR frozen ResNet-50
# ============================================================

print(
    "\n========================================"
)

print(
    "Loading SimCLR pretrained encoder"
)

print(
    "========================================\n"
)


simclr_model = SimCLR(
    projection_dim=128,
    projection_hidden_dim=2048,
).to(device)


checkpoint = torch.load(
    simclr_checkpoint_path,
    map_location=device,
    weights_only=False,
)


simclr_model.load_state_dict(
    checkpoint["model_state_dict"]
)


simclr_encoder = simclr_model.encoder


for parameter in simclr_encoder.parameters():
    parameter.requires_grad = False


simclr_train_features, _ = (
    extract_features(
        simclr_encoder,
        train_loader,
        "SimCLR encoder - train",
    )
)


simclr_test_features, _ = (
    extract_features(
        simclr_encoder,
        test_loader,
        "SimCLR encoder - test",
    )
)


print(
    "SimCLR train features:",
    simclr_train_features.shape,
)

print(
    "SimCLR test features:",
    simclr_test_features.shape,
)


simclr_accuracy = train_linear_classifier(
    simclr_train_features,
    train_labels,
    simclr_test_features,
    test_labels,
    "SimCLR frozen ResNet-50",
)


# ============================================================
# Final comparison
# ============================================================

print(
    "\n========================================"
)

print(
    "FINAL LINEAR PROBE RESULTS"
)

print(
    "========================================"
)


print(
    f"Random frozen ResNet-50 : "
    f"{random_accuracy:.2f}%"
)

print(
    f"SimCLR frozen ResNet-50 : "
    f"{simclr_accuracy:.2f}%"
)

print(
    f"Improvement              : "
    f"{simclr_accuracy - random_accuracy:.2f} "
    f"percentage points"
)

print(
    "========================================\n"
)

import json

results = {
    "random_frozen_resnet50_accuracy": random_accuracy,
    "simclr_frozen_resnet50_accuracy": simclr_accuracy,
    "improvement_percentage_points": simclr_accuracy - random_accuracy,
}

results_path = "/tmp/linear_probe_results.json"

with open(results_path, "w") as f:
    json.dump(results, f, indent=4)

subprocess.run(
    [
        "aws", "s3", "cp",
        results_path,
        (
            "s3://lachqar/"
            "self-supervised-visual-representation-learning/"
            "results/linear_probe_food101.json"
        ),
        "--endpoint-url",
        endpoint,
    ],
    check=True,
)