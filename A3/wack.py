from typing import Tuple
import datetime

import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class Args:
    """Command-line arguments to store model configuration."""

    num_classes = 10

    # Hyperparameters
    epochs = 25  # Should easily reach above 65% test acc after 20 epochs with a hidden_size of 64
    batch_size = 128
    lr = 1e-3
    weight_decay = 1e-4

    # Hyperparameters for ViT
    input_resolution = 32
    in_channels = 3
    patch_size = 4
    hidden_size = 64
    layers = 6
    heads = 8

    # Save your model as "vit-cifar10-{YOUR_CCID}"
    YOUR_CCID = "123456"  # Replace "123456" with your actual CCID
    name = f"vit-cifar10-{YOUR_CCID}"


class PatchEmbeddings(nn.Module):
    """Compute patch embedding of shape `(batch_size, seq_length, hidden_size)`."""

    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        hidden_size: int,
        in_channels: int = 3,  # 3 for RGB, 1 for Grayscale
    ):
        super().__init__()
        self.patch_size = patch_size
        self.input_resolution = input_resolution
        self.num_patches = (input_resolution // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # x: (batch_size, in_channels, H, W)
        x = self.projection(
            x
        )  # (batch_size, hidden_size, num_patches ** 0.5, num_patches ** 0.5)
        x = x.flatten(2)  # (batch_size, hidden_size, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, hidden_size)
        return x


class PositionEmbedding(nn.Module):
    """Calculate position embeddings with [CLS] and [POS]."""

    def __init__(
        self,
        num_patches: int,
        hidden_size: int,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches + 1, hidden_size)
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings: (batch_size, num_patches, hidden_size)
        batch_size = embeddings.size(0)
        cls_tokens = self.cls_token.expand(
            batch_size, -1, -1
        )  # (batch_size, 1, hidden_size)
        embeddings = torch.cat(
            (cls_tokens, embeddings), dim=1
        )  # (batch_size, num_patches +1, hidden_size)
        embeddings = embeddings + self.position_embeddings  # Add positional embeddings
        return embeddings


class TransformerEncoderBlock(nn.Module):
    """A residual Transformer encoder block."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_head, batch_first=True
        )
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        attn_output, _ = self.attn(x, x, x)  # Self-attention
        x = x + attn_output  # Residual connection
        x = self.ln_1(x)  # Layer normalization
        mlp_output = self.mlp(x)  # MLP
        x = x + mlp_output  # Residual connection
        x = self.ln_2(x)  # Layer normalization
        return x


class ViT(nn.Module):
    """Vision Transformer."""

    def __init__(
        self,
        num_classes: int,
        input_resolution: int,
        patch_size: int,
        in_channels: int,
        hidden_size: int,
        layers: int,
        heads: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_embed = PatchEmbeddings(
            input_resolution, patch_size, hidden_size, in_channels
        )
        num_patches = (input_resolution // patch_size) ** 2
        self.pos_embed = PositionEmbedding(num_patches, hidden_size)
        self.ln_pre = nn.LayerNorm(hidden_size)
        self.transformer = nn.Sequential(
            *[TransformerEncoderBlock(hidden_size, heads) for _ in range(layers)]
        )
        self.ln_post = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, in_channels, H, W)
        x = self.patch_embed(x)  # (batch_size, num_patches, hidden_size)
        x = self.pos_embed(x)  # (batch_size, num_patches +1, hidden_size)
        x = self.ln_pre(x)  # Layer normalization
        x = self.transformer(x)  # (batch_size, num_patches +1, hidden_size)
        x = self.ln_post(x)  # Layer normalization
        cls_token = x[:, 0]  # (batch_size, hidden_size)
        x = self.classifier(cls_token)  # (batch_size, num_classes)
        return x


def transform(
    input_resolution: int,
    mode: str = "train",
    mean: Tuple[float] = (0.5, 0.5, 0.5),  # Modify as needed
    std: Tuple[float] = (0.5, 0.5, 0.5),  # Modify as needed
):
    """Preprocess the image inputs with at least 3 data augmentations for training."""
    if mode == "train":
        tfm = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(input_resolution, padding=4),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        tfm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    return tfm


def inverse_transform(
    img_tensor: torch.Tensor,
    mean: Tuple[float] = (-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5),  # Modify as needed
    std: Tuple[float] = (1 / 0.5, 1 / 0.5, 1 / 0.5),  # Modify as needed
) -> np.ndarray:
    """Revert the normalization process and convert the tensor back to a numpy image."""
    inv_normalize = transforms.Normalize(mean=mean, std=std)
    img_tensor = inv_normalize(img_tensor).clamp(0, 1).permute(1, 2, 0)
    img = np.uint8(255 * img_tensor.cpu().numpy())
    return img


def train_vit_model(args):
    """Train loop for ViT model."""
    # Dataset for train / test
    tfm_train = transform(
        input_resolution=args.input_resolution,
        mode="train",
    )

    tfm_test = transform(
        input_resolution=args.input_resolution,
        mode="test",
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=tfm_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=tfm_test
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Define ViT model
    model = ViT(
        num_classes=args.num_classes,
        input_resolution=args.input_resolution,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        hidden_size=args.hidden_size,
        layers=args.layers,
        heads=args.heads,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss, optimizer and lr scheduler
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training loop
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1} / {args.epochs}")

        epoch_loss = 0.0
        for i, (x, labels) in enumerate(pbar):
            x = x.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        # Compute average loss for the epoch
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Average Loss: {avg_loss:.4f}")

        # Evaluate at the end of the epoch
        test_acc = test_classification_model(model, test_loader, device)

        # Save the model if it has the best accuracy so far
        if test_acc > best_acc:
            best_acc = test_acc
            state_dict = {
                "model": model.state_dict(),
                "acc": best_acc,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            torch.save(state_dict, f"{args.name}.pt")
            print("Best test acc:", best_acc)
        else:
            print("Test acc:", test_acc)
        print()


def test_classification_model(
    model: nn.Module,
    test_loader,
    device: torch.device,
):
    """Evaluate the model on the test set."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
