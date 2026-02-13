import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import os
from tqdm import tqdm
import random
import torchio as tio
from models.unet_3d import UNet3D


# used for unet
# class PatchDataset(Dataset):
#     def __init__(self, cases, images_dir, labels_dir, patches_per_case=1, patch_size=96, augment=False, foreground_prob=0.5):
#         self.cases = cases
#         self.images_dir = images_dir
#         self.labels_dir = labels_dir
#         self.patch_size = patch_size
#         self.patches_per_case = patches_per_case
#         self.augment = augment
#         self.foreground_prob = foreground_prob 

#         if augment:
#             self.transform = tio.Compose([
#                 tio.RandomFlip(axes=('LR',)),
#                 tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
#                 tio.RandomNoise(mean=0, std=0.01),
#                 tio.RandomGamma(log_gamma=(-0.3, 0.3))
#             ])

#     def __len__(self):
#         return len(self.cases) * self.patches_per_case

#     def __getitem__(self, idx):
#         case = self.cases[idx // self.patches_per_case]

#         image = nib.load(os.path.join(self.images_dir, f"{case}.nii.gz")).get_fdata()
#         label = nib.load(os.path.join(self.labels_dir, f"{case}.nii.gz")).get_fdata()

#         z, y, x = image.shape
#         ps = self.patch_size

#         # --- KEY FIX: Forced Foreground Sampling ---
#         # 50% chance to force the patch to be centered on an organ
#         if random.random() < self.foreground_prob:
#             # Find all coordinates that are NOT background
#             fg_indices = np.argwhere(label > 0)
            
#             if len(fg_indices) > 0:
#                 # Pick a random foreground voxel
#                 center = fg_indices[random.randint(0, len(fg_indices) - 1)]
                
#                 # Calculate top-left corner (z0, y0, x0) to center the patch on that voxel
#                 z0 = int(np.clip(center[0] - ps // 2, 0, z - ps))
#                 y0 = int(np.clip(center[1] - ps // 2, 0, y - ps))
#                 x0 = int(np.clip(center[2] - ps // 2, 0, x - ps))
#             else:
#                 # Fallback if the volume is empty (rare)
#                 z0 = random.randint(0, z - ps)
#                 y0 = random.randint(0, y - ps)
#                 x0 = random.randint(0, x - ps)
#         else:
#             # Standard random crop
#             z0 = random.randint(0, z - ps)
#             y0 = random.randint(0, y - ps)
#             x0 = random.randint(0, x - ps)
#         # ---

#         image_patch = image[z0:z0+ps, y0:y0+ps, x0:x0+ps]
#         label_patch = label[z0:z0+ps, y0:y0+ps, x0:x0+ps]

#         image_patch = torch.tensor(image_patch, dtype=torch.float32).unsqueeze(0)
#         label_patch = torch.tensor(label_patch, dtype=torch.long)

#         if self.augment:
#             subject = tio.Subject(
#                 image=tio.ScalarImage(tensor=image_patch),
#                 label=tio.LabelMap(tensor=label_patch.unsqueeze(0))
#             )
#             subject = self.transform(subject)
#             image_patch = subject.image.data
#             label_patch = subject.label.data.squeeze(0).long()

#         return image_patch, label_patch

class PatchDataset(Dataset):
    def __init__(
        self,
        cases,
        images_dir,
        labels_dir,
        patches_per_case=1,
        patch_size=96,
        augment=False,
        foreground_prob=0.5,
    ):
        self.cases = cases
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.patch_size = patch_size
        self.patches_per_case = patches_per_case
        self.augment = augment
        self.foreground_prob = foreground_prob

        if augment:
            self.transform = tio.Compose([
                tio.RandomFlip(axes=('LR',)),
                tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
                tio.RandomNoise(mean=0, std=0.01),
                tio.RandomGamma(log_gamma=(-0.3, 0.3))
            ])

    def __len__(self):
        return len(self.cases) * self.patches_per_case

    def __getitem__(self, idx):

        case = self.cases[idx // self.patches_per_case]

        image = nib.load(os.path.join(self.images_dir, f"{case}.nii.gz")).get_fdata(dtype=np.float32)
        label = nib.load(os.path.join(self.labels_dir, f"{case}.nii.gz")).get_fdata()
        label = label.astype(np.uint8)



        z, y, x = image.shape
        ps = self.patch_size

        # ------ Balanced Sampling ------
        if random.random() < self.foreground_prob:

            # Get all foreground classes present in this case
            classes = np.unique(label)
            classes = classes[classes != 0]  # remove background

            if len(classes) > 0:

                # Randomly choose ONE class (balanced)
                chosen_class = random.choice(classes.tolist())

                class_voxels = np.argwhere(label == chosen_class)

                if len(class_voxels) > 0:
                    center = class_voxels[random.randint(0, len(class_voxels) - 1)]

                    z0 = int(np.clip(center[0] - ps // 2, 0, z - ps))
                    y0 = int(np.clip(center[1] - ps // 2, 0, y - ps))
                    x0 = int(np.clip(center[2] - ps // 2, 0, x - ps))
                else:
                    # fallback random crop
                    z0 = random.randint(0, z - ps)
                    y0 = random.randint(0, y - ps)
                    x0 = random.randint(0, x - ps)
            else:
                # no foreground in volume (rare)
                z0 = random.randint(0, z - ps)
                y0 = random.randint(0, y - ps)
                x0 = random.randint(0, x - ps)

        else:
            # ------ Random Crop ------
            z0 = random.randint(0, z - ps)
            y0 = random.randint(0, y - ps)
            x0 = random.randint(0, x - ps)
        # 

        image_patch = image[z0:z0+ps, y0:y0+ps, x0:x0+ps]
        label_patch = label[z0:z0+ps, y0:y0+ps, x0:x0+ps]

        image_patch = torch.tensor(image_patch, dtype=torch.float32).unsqueeze(0)
        label_patch = torch.tensor(label_patch, dtype=torch.long)

        if self.augment:
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image_patch),
                label=tio.LabelMap(tensor=label_patch.unsqueeze(0))
            )
            subject = self.transform(subject)
            image_patch = subject.image.data
            label_patch = subject.label.data.squeeze(0).long()

        return image_patch, label_patch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dice_loss(pred, target, num_classes=7 ,smooth=1e-5):
    pred = torch.softmax(pred, dim=1)
    target = target.long()

    target_onehot = torch.nn.functional.one_hot(target, num_classes=num_classes)
    target_onehot = target_onehot.permute(0,4,1,2,3).float()

    intersection = (pred * target_onehot).sum(dim=(2,3,4))
    union = pred.sum(dim=(2,3,4)) + target_onehot.sum(dim=(2,3,4))

    dice = (2 * intersection + smooth) / (union + smooth)

    dice = dice[:, 1:]  

    return 1 - dice.mean()




#  Validation Dice Metric (Per Organ)
def compute_per_class_dice(pred, target, num_classes=7, smooth=1e-5):
    """
    Returns Dice for each class (including background)
    """

    pred = torch.argmax(pred, dim=1)

    dices = []

    for cls in range(1, num_classes):  # ignore background
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()

        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()

        dice = (2. * intersection + smooth) / (union + smooth)
        dices.append(dice.item())

    return dices


#  Training Function (One Epoch)
def train_one_epoch(model, loader, optimizer, scaler, ce_loss, device, num_classes=7):

    model.train()
    total_loss = 0

    for images, labels in tqdm(loader):

        images = images.to(device)
        labels = labels.long().to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = dice_loss(outputs, labels, num_classes) + ce_loss(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


#  Validation Function
def validate_one_epoch(model, loader, ce_loss, device, num_classes=7):

    model.eval()
    total_loss = 0
    all_dices = []

    with torch.no_grad():
        for images, labels in loader:

            images = images.to(device)
            labels = labels.long().to(device)

            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = dice_loss(outputs, labels, num_classes) + ce_loss(outputs, labels)

            total_loss += loss.item()

            batch_dice = compute_per_class_dice(outputs, labels, num_classes)
            all_dices.append(batch_dice)

    mean_loss = total_loss / len(loader)

    # Average dice per class
    all_dices = np.array(all_dices)
    mean_dices = np.mean(all_dices, axis=0)

    return mean_loss, mean_dices



#  Checkpoint Utilities
def save_checkpoint(model, optimizer, epoch, best_val_loss, save_path):

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss
    }, save_path)


def load_checkpoint(model, optimizer, load_path, device):

    checkpoint = torch.load(load_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    best_val_loss = checkpoint["best_val_loss"]

    return model, optimizer, epoch, best_val_loss





def sliding_window_inference(model, image, patch_size=80, stride=60, device="cuda"):

    model.eval()
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    _, _, D, H, W = image.shape
    num_classes = 7

    output = torch.zeros((1, num_classes, D, H, W), dtype=torch.float32)
    count_map = torch.zeros_like(output)

    with torch.no_grad():

        z_steps = list(range(0, D - patch_size, stride)) + [D - patch_size]
        y_steps = list(range(0, H - patch_size, stride)) + [H - patch_size]
        x_steps = list(range(0, W - patch_size, stride)) + [W - patch_size]

        for z in z_steps:
            for y in y_steps:
                for x in x_steps:

                    patch = image[:, :, z:z+patch_size, y:y+patch_size, x:x+patch_size]

                    logits = model(patch).cpu()

                    output[:, :, z:z+patch_size, y:y+patch_size, x:x+patch_size] += logits
                    count_map[:, :, z:z+patch_size, y:y+patch_size, x:x+patch_size] += 1

    count_map[count_map == 0] = 1
    output = output / count_map

    # Apply softmax AFTER averaging logits
    output = torch.softmax(output, dim=1)

    output = torch.argmax(output, dim=1)

    return output.squeeze().cpu().numpy()



def evaluate_full_volume(model, cases, images_dir, labels_dir, device="cuda"):

    model.eval()
    all_dices = []

    for case in cases:

        image = nib.load(f"{images_dir}/{case}.nii.gz").get_fdata()
        label = nib.load(f"{labels_dir}/{case}.nii.gz").get_fdata()
        label = label.astype(np.uint8)

        pred = sliding_window_inference(model, image, device=device)
        print("Unique pred labels:", np.unique(pred))
        print("Unique GT labels:", np.unique(label))


        case_dices = []

        for cls in range(1, 7):  # ignore background

            pred_cls = (pred == cls)
            label_cls = (label == cls)

            intersection = np.sum(pred_cls & label_cls)
            union = np.sum(pred_cls) + np.sum(label_cls)

            dice = (2 * intersection + 1e-5) / (union + 1e-5)
            case_dices.append(dice)

        all_dices.append(case_dices)

        print(f"{case} Dice:", case_dices)

    all_dices = np.array(all_dices)

    mean_dice = np.mean(all_dices, axis=0)
    std_dice = np.std(all_dices, axis=0)

    print("\nMean Dice per organ:", mean_dice)
    print("Std Dice per organ:", std_dice)

    return mean_dice, std_dice


import matplotlib.pyplot as plt
import numpy as np

def show_difference_map(ct_slice, gt_slice, pred_slice, organ_id=None):
    """
    organ_id: if None → full multi-class comparison
              if integer → show only that organ
    """

    if organ_id is not None:
        gt_mask = (gt_slice == organ_id)
        pred_mask = (pred_slice == organ_id)
    else:
        gt_mask = gt_slice > 0
        pred_mask = pred_slice > 0

    true_positive = gt_mask & pred_mask
    false_positive = (~gt_mask) & pred_mask
    false_negative = gt_mask & (~pred_mask)

    diff_map = np.zeros((*gt_slice.shape, 3))

    # Green = correct
    diff_map[true_positive] = [0, 1, 0]

    # Red = false positive
    diff_map[false_positive] = [1, 0, 0]

    # Blue = false negative
    diff_map[false_negative] = [0, 0, 1]

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(ct_slice, cmap='gray')
    plt.title("CT Slice")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(ct_slice, cmap='gray')
    plt.imshow(diff_map, alpha=0.6)
    plt.title("Difference Map\nGreen=Correct, Red=FP, Blue=FN")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def show_organ_overlay(ct_slice, gt_slice, pred_slice, organ_id=None):

    plt.figure(figsize=(12,5))

    #  Ground Truth 
    plt.subplot(1,2,1)
    plt.imshow(ct_slice, cmap='gray')

    if organ_id is not None:
        gt_mask = (gt_slice == organ_id)
        plt.imshow(gt_mask, cmap='jet', alpha=0.5)
        plt.title(f"Ground Truth - Organ {organ_id}")
    else:
        plt.imshow(gt_slice, cmap='tab20', alpha=0.5)
        plt.title("Ground Truth - Multi Organ")

    plt.axis("off")

    #  Prediction 
    plt.subplot(1,2,2)
    plt.imshow(ct_slice, cmap='gray')

    if organ_id is not None:
        pred_mask = (pred_slice == organ_id)
        plt.imshow(pred_mask, cmap='jet', alpha=0.5)
        plt.title(f"Prediction - Organ {organ_id}")
    else:
        plt.imshow(pred_slice, cmap='tab20', alpha=0.5)
        plt.title("Prediction - Multi Organ")

    plt.axis("off")

    plt.tight_layout()
    plt.show()

