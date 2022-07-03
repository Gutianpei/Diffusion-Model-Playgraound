from torch.utils.data import Dataset
import lmdb
from io import BytesIO
from PIL import Image
import torchvision.transforms as tfs
import os
import pandas as pd

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256, attr_path=None):
        # self.env = lmdb.open(
        #     path,
        #     max_readers=32,
        #     readonly=True,
        #     lock=False,
        #     readahead=False,
        #     meminit=False,
        # )

        # if not self.env:
        #     raise IOError("Cannot open lmdb dataset", path)

        if not os.path.exists(path):
            raise IOError("the path doesn't exist", path)

        self.data_path = path
        self.files = sorted(os.listdir(path))
        self.length = len(self.files)
        if attr_path:
            #print(attr_path)
            self.attr_df = pd.read_csv(attr_path)


        # with self.env.begin(write=False) as txn:
        #     self.length = int(txn.get("length".encode("utf-8")).decode("utf-8"))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # with self.env.begin(write=False) as txn:
        #     key = f"{self.resolution}-{str(index).zfill(5)}".encode("utf-8")
        #     img_bytes = txn.get(key)
        # buffer = BytesIO(img_bytes)
        # img = Image.open(buffer)

        img = Image.open(os.path.join(self.data_path, self.files[index])).resize((self.resolution, self.resolution))
        img = self.transform(img)
        attr = self.attr_df.iloc[index]
        attr = [attr["Young"], attr["Male"], attr["Eyeglasses"]]
        return img, attr


################################################################################

def get_celeba_dataset(data_root, config):
    train_transform = tfs.Compose([tfs.ToTensor(),
                                   tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                                 inplace=True)])

    val_transform = tfs.Compose([tfs.ToTensor(),
                                   tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                                 inplace=True)])

    test_transform = tfs.Compose([tfs.ToTensor(),
                                  tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                                inplace=True)])

    train_dataset = MultiResolutionDataset(os.path.join(data_root, 'raw_images', 'train', 'images'),
                                            train_transform, config.data.image_size,
                                            os.path.join(data_root, 'list_attr_celeba.csv'))
    val_dataset = MultiResolutionDataset(os.path.join(data_root, 'raw_images', 'val', 'images'),
                                            val_transform, config.data.image_size,
                                            os.path.join(data_root, 'list_attr_celeba.csv'))
    test_dataset = MultiResolutionDataset(os.path.join(data_root, 'raw_images', 'train', 'images'),
                                            test_transform, config.data.image_size,
                                            os.path.join(data_root, 'list_attr_celeba.csv'))
    # train_dataset = MultiResolutionDataset(os.path.join(data_root, 'raw_images', 'train', 'images'),
    #                                         train_transform, 256,
    #                                         os.path.join(data_root, 'list_attr_celeba.csv'))
    # val_dataset = MultiResolutionDataset(os.path.join(data_root, 'raw_images', 'val', 'images'),
    #                                         val_transform, 256,
    #                                         os.path.join(data_root, 'list_attr_celeba.csv'))
    # test_dataset = MultiResolutionDataset(os.path.join(data_root, 'raw_images', 'train', 'images'),
    #                                         test_transform, 256,
    #                                         os.path.join(data_root, 'list_attr_celeba.csv'))


    return train_dataset, val_dataset, test_dataset
