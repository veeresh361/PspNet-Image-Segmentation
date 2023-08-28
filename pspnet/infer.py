import os
import time
import pandas as pd
from config import (
    TRAIN_IMAGE_PATH,
    TRAIN_MASK_PATH,
    TEST_NG_PATH,
    TEST_OK_PATH,
    TEST_OK_MASK_PATH,
    TEST_NG_MASK_PATH,
)
from utils import (
    get_train_augs,
    get_valid_augs,
    validate_and_save_predictions,
    getSegmentedOutputs,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from loader import SegmentationDataset, TestSegmentationDataset


test_image_filenames = os.listdir(TRAIN_IMAGE_PATH)
test_image_filenames = [
    os.path.join(TRAIN_IMAGE_PATH, filename) for filename in test_image_filenames
]

test_mask_filenames = os.listdir(TRAIN_MASK_PATH)
test_mask_filenames = [
    os.path.join(TRAIN_MASK_PATH, filename) for filename in test_mask_filenames
]

test_df = pd.DataFrame({"images": test_image_filenames, "masks": test_mask_filenames})


valid_set = SegmentationDataset(test_df, get_valid_augs())
valid_loader = DataLoader(valid_set, batch_size=1)

begin = time.time()
validate_and_save_predictions(valid_loader, test_df)
# getSegmentedOutputs(ng_loader,ng_df)
time.sleep(1)
end = time.time()
print(f"Total runtime of the program is {end - begin}")
