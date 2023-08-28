import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import torch
from config import (
    IMG_SIZE,
    DEVICE,
    EPOCHS,
    SAVE_MODEL_PATH,
    LR,
    PREDICTED_BINARY_MASK_PATH,
    SEGMENTED_OUTPUT_PATH,
    SEGMENTED_COMBINED_IMAGE_PATH,
    ACTUAL_IMAGE_PATH,
    ACTUAL_BINARY_MASK_PATH,
    TEST_NG_MASK_PATH,
    TEST_OK_MASK_PATH,
)
import albumentations as A
from PSPNet import SegmentationModel
from torchvision.utils import save_image


def get_train_augs():
    return A.Compose(
        [A.Resize(IMG_SIZE, IMG_SIZE), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]
    )


def get_valid_augs():
    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
        ]
    )


def predict_image(model, image):
    model.eval()
    with torch.no_grad():
        logits_mask = model(image.to(DEVICE))
        pred_mask = torch.sigmoid(logits_mask)
        pred_mask = (pred_mask > 0.5).float()
    return pred_mask.squeeze(0).cpu().numpy()


def calculate_precision_recall(pred_mask, true_mask):
    TP = np.logical_and(pred_mask, true_mask).sum()
    FP = np.logical_and(pred_mask, 1 - true_mask).sum()
    FN = np.logical_and(1 - pred_mask, true_mask).sum()

    precision = TP / (
        TP + FP + 1e-6
    )  # Adding a small epsilon to avoid division by zero
    recall = TP / (TP + FN + 1e-6)

    return precision, recall


def calculate_iou(pred_mask, true_mask):
    # print(pred_mask.shape)
    # print(true_mask.shape)
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()

    iou = intersection / (union + 1e-8)  # Add a small epsilon to avoid division by zero
    return iou


def train_fn(dataloader, model, optimizer):
    model.train()
    total_loss = 0.0

    with tqdm(
        total=len(dataloader), desc="Training", unit="batch", dynamic_ncols=True
    ) as progress:
        for images, masks in dataloader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            logits, loss = model(images, masks)
            loss.backward()  # gradient of params
            optimizer.step()  # update params

            total_loss += loss.item()
            progress.set_postfix(loss=total_loss / (progress.n + 1))
            progress.update()

    return total_loss / len(dataloader)


# Validation function
def eval_fn(dataloader, model):
    model.eval()
    total_loss = 0.0

    with torch.no_grad(), tqdm(
        total=len(dataloader), desc="Validating", unit="batch", dynamic_ncols=True
    ) as progress:
        for images, masks in dataloader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits, loss = model(images, masks)

            total_loss += loss.item()
            progress.set_postfix(loss=total_loss / (progress.n + 1))
            progress.update()

    return total_loss / len(dataloader)


def validate_and_save_predictions(valid_set, valid_Dataset):
    model = SegmentationModel().to(DEVICE)
    model.load_state_dict(torch.load(SAVE_MODEL_PATH + "best-model_350.pt"))

    # if not os.path.exists('Validation'):
    #     os.makedirs('Validation')
    iocList = []
    precision = []
    recall = []
    with torch.no_grad(), tqdm(
        total=len(valid_set), desc="Validating", unit="image", dynamic_ncols=True
    ) as progress:
        for index, (image, mask) in enumerate(valid_set):
            image = image.to(DEVICE)
            imageName = valid_Dataset.iloc[index][0].split("/")[-1]
            save_image(image, ACTUAL_IMAGE_PATH + imageName)
            newimage = cv2.imread(ACTUAL_IMAGE_PATH + imageName, 0)
            pred_mask = predict_image(model, image)
            pred_mask = pred_mask.reshape(IMG_SIZE, IMG_SIZE)
            mask = mask.cpu().numpy().reshape(IMG_SIZE, IMG_SIZE)
            plt.imsave(ACTUAL_BINARY_MASK_PATH + imageName, mask, cmap="gray")
            pred_mask_binary = (pred_mask > 0.5).astype(np.uint8)
            plt.imsave(
                PREDICTED_BINARY_MASK_PATH + imageName, pred_mask_binary, cmap="gray"
            )
            contours, _ = cv2.findContours(
                pred_mask_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )

            # Draw contours on the original image
            try:
                image_with_contours = newimage.copy()
                cv2.drawContours(image_with_contours, contours, -1, (255, 0, 0), 2)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                largest_contour = contours[0]
                lastImage = cv2.bitwise_and(newimage, newimage, mask=pred_mask_binary)
                x, y, w, h = cv2.boundingRect(largest_contour)
                cropped_image = lastImage[y : y + h, x : x + w]
                plt.imsave(
                    SEGMENTED_OUTPUT_PATH + imageName, cropped_image, cmap="gray"
                )

                fig, axes = plt.subplots(1, 5, figsize=(15, 5))

                # Plot original image
                axes[0].imshow(newimage, cmap="gray")
                axes[0].set_title("Original Image")

                # Plot ground truth mask
                axes[1].imshow(mask, cmap="gray")
                axes[1].set_title("Ground Truth Mask")

                # Plot predicted mask
                axes[2].imshow(pred_mask, cmap="gray")
                axes[2].set_title("Predicted Mask")

                # Plot original image with contours
                axes[3].imshow(image_with_contours, cmap="gray")
                axes[3].set_title("Image with countour")

                axes[4].imshow(lastImage, cmap="gray")
                axes[4].set_title("Segmented Image")

                plt.savefig(SEGMENTED_COMBINED_IMAGE_PATH + imageName)
                plt.close()
            except Exception as e:
                print(e)


def getSegmentedOutputs(valid_set, valid_Dataset):
    model = SegmentationModel().to(DEVICE)
    model.load_state_dict(torch.load(SAVE_MODEL_PATH + "best-model_100.pt"))
    with torch.no_grad(), tqdm(
        total=len(valid_set), desc="Validating", unit="image", dynamic_ncols=True
    ) as progress:
        for index, (image) in enumerate(valid_set):
            image = image.to(DEVICE)
            imageName = valid_Dataset.iloc[index][0].split("/")[-1]
            save_image(image, ACTUAL_IMAGE_PATH + imageName)
            newimage = cv2.imread(ACTUAL_IMAGE_PATH + imageName, 0)
            pred_mask = predict_image(model, image)
            pred_mask = pred_mask.reshape(320, 320)
            pred_mask_binary = (pred_mask > 0.5).astype(np.uint8)
            plt.imsave(PREDICTED_BINARY_MASK_PATH + imageName, pred_mask, cmap="gray")

            contours, _ = cv2.findContours(
                pred_mask_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )

            # Draw contours on the original image
            try:
                image_with_contours = newimage.copy()
                cv2.drawContours(image_with_contours, contours, -1, (255, 0, 0), 2)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                largest_contour = contours[0]
                lastImage = cv2.bitwise_and(newimage, newimage, mask=pred_mask_binary)
                x, y, w, h = cv2.boundingRect(largest_contour)
                cropped_image = lastImage[y : y + h, x : x + w]
                plt.imsave(
                    SEGMENTED_OUTPUT_PATH + imageName, cropped_image, cmap="gray"
                )
                fig, axes = plt.subplots(1, 4, figsize=(15, 5))

                # Plot original image
                axes[0].imshow(newimage, cmap="gray")
                axes[0].set_title("Original Image")

                # Plot predicted mask
                axes[1].imshow(pred_mask, cmap="gray")
                axes[1].set_title("Predicted Mask")

                # Plot original image with contours
                axes[2].imshow(image_with_contours, cmap="gray")
                axes[2].set_title("Image with countour")

                axes[3].imshow(lastImage, cmap="gray")
                axes[3].set_title("Segmented Image")

                plt.savefig(SEGMENTED_COMBINED_IMAGE_PATH + imageName)
                plt.close()
            except Exception as e:
                pass


def train_and_save_model(train_loader, valid_loader):
    model = SegmentationModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_loss = np.Inf
    trainLoss = []
    validLoss = []
    for epoch in range(EPOCHS):
        train_loss = train_fn(train_loader, model, optimizer)
        valid_loss = eval_fn(valid_loader, model)
        trainLoss.append(train_loss)
        validLoss.append(valid_loss)

        if valid_loss < best_val_loss:
            torch.save(
                model.state_dict(),
                SAVE_MODEL_PATH + "best-model_" + str(EPOCHS) + ".pt",
            )
            best_val_loss = valid_loss

        print(f"Epoch: [{epoch + 1}] train_loss: {train_loss} valid_loss: {valid_loss}")
    plt.plot(list(range(EPOCHS)), trainLoss, label="Training Loss")
    plt.plot(list(range(EPOCHS)), validLoss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("/home/veeresh/segmentation/pspnet/graph.jpg")
