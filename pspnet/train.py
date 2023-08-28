import os
import time
import pandas as pd
from config import TRAIN_IMAGE_PATH, TRAIN_MASK_PATH
from utils import get_train_augs, get_valid_augs, train_and_save_model
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from loader import SegmentationDataset


image_filenames = os.listdir(TRAIN_IMAGE_PATH)
image_filenames = [
    os.path.join(TRAIN_IMAGE_PATH, filename) for filename in image_filenames
]
mask_filenames = os.listdir(TRAIN_MASK_PATH)
mask_filenames = [
    os.path.join(TRAIN_MASK_PATH, filename) for filename in mask_filenames
]

df = pd.DataFrame({"images": image_filenames, "masks": mask_filenames})

train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
train_set = SegmentationDataset(train_df, get_train_augs())
valid_set = SegmentationDataset(valid_df, get_valid_augs())

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=32)

begin = time.time()
train_and_save_model(train_loader, valid_loader)
time.sleep(1)
end = time.time()
print(f"Total runtime of the program is {end - begin}")
