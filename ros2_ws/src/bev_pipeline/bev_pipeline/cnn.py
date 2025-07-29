import os
import glob

import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# 1) Dataset
class BEVDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Loading as NumPy arrays
        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_COLOR) # H x W x 3 (BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE) # H x W

        # Augmentation
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']

        # Converting to torch.Tensor
        # img: H x W x 3 --> 3 x H x W
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        # mask: H x W --> 1 x H x W
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        return img_tensor, mask_tensor


# 2) Model
class UNetTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3,8,3,padding=1), nn.ReLU(),
            nn.Conv2d(8,16,3,padding=1), nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(16,8,2,2), nn.ReLU(),
            nn.Conv2d(8,1,1)
        )
    def forward(self,x):
        return self.dec(self.enc(x))


# 3) Training Loop
def train(
    img_dir = "dataset/bev_images",
    mask_dir = "dataset/bev_masks",
    epochs = 10,
    batch_size = 4,
    lr = 1e-3,
    val_split = 0.2
):
    
    full_dataset = BEVDataset(img_dir, mask_dir)

    # Calculating split sizes
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size

    # Splitting the dataset
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Dataset size: {dataset_size}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")


    # DataLoaders
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model, optimizer, loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetTiny().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn= nn.BCEWithLogitsLoss()

    # Variables to track best model for saving
    best_val_loss = float('inf')
    best_epoch = 0

    # Training
    for epoch in range(1, epochs + 1):
        model.train()
        train_total_loss, train_total_acc = 0.0, 0.0

        for imgs, masks in train_dl:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, masks)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_total_loss += loss.item()
            with torch.no_grad():
                acc = ((preds.sigmoid() > 0.5) == (masks > 0.5)).float().mean().item()
                train_total_acc += acc

        avg_train_loss = train_total_loss / len(train_dl)
        avg_train_acc = train_total_acc / len(train_dl)

        # Validation Phase
        model.eval() # Set model to evaluation mode
        val_total_loss, val_total_acc = 0.0, 0.0

        with torch.no_grad(): 
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                loss = loss_fn(preds, masks)

                val_total_loss += loss.item()
                acc = ((preds.sigmoid() > 0.5) == (masks > 0.5)).float().mean().item()
                val_total_acc += acc

        avg_val_loss = val_total_loss / len(val_dl)
        avg_val_acc = val_total_acc / len(val_dl)

        print(f"Epoch {epoch}/{epochs} — Train Loss: {avg_train_loss:.3f}, Train Acc: {avg_train_acc:.3f} | Val Loss: {avg_val_loss:.3f}, Val Acc: {avg_val_acc:.3f}")

        # Saving the model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            os.makedirs("models", exist_ok=True) # Ensure 'models' directory exists
            torch.save(model.state_dict(), "models/bev_model_best_val.pth")
            print(f" => New best model saved at Epoch {epoch} with Val Loss: {best_val_loss:.3f}")

    print("Training complete")
    print(f"Best model saved at models/bev_model_best_val.pth from Epoch {best_epoch} with Validation Loss: {best_val_loss:.3f}")


if __name__ == "__main__":
    train()