import pathlib
import cv2
import numpy as np


class Dataset:
    def __init__(self, dir):
        self.images_train = list(pathlib.Path(f"{dir}/images/train").iterdir())
        self.images_val = list(pathlib.Path(f"{dir}/images/val").iterdir())
        self.images_test = list(pathlib.Path(f"{dir}/images/test").iterdir())
        
        self.labels_train = list(pathlib.Path(f"{dir}/labels/train").iterdir())
        self.labels_val = list(pathlib.Path(f"{dir}/labels/val").iterdir())
        self.labels_test = list(pathlib.Path(f"{dir}/labels/test").iterdir())

        self.length_train = None
        self.length_val = None
        self.length_test = None

        self.__len__()
        self.__unpack__()
        
    def __len__(self):
        # костиль
        if len(self.images_train) == len(self.labels_train):
            self.length_train = len(self.images_train)
        
        if len(self.images_val) == len(self.labels_val):
            self.length_val = len(self.images_val)

        if len(self.images_test) == len(self.labels_test):
            self.length_test = len(self.images_test)
        
        for i, x in enumerate([self.length_train, self.length_val, self.length_test]):
            if x == None:
                raise ValueError(f"Mismatch number of images/labels.")
    
    def __unpack__(self):
        self.images_train = [cv2.resize(cv2.imdecode(np.fromfile(x, dtype=np.uint8), cv2.IMREAD_COLOR), (640, 640)) for x in self.images_train]
        
    
class Dataloader:
    def __init__(self, dataset, batch_size):
        self.data = dataset
        self.batch_size = batch_size
        
        self.num_batches = {
            "train": int(np.ceil(self.data.length_train / self.batch_size)),
            "val": int(np.ceil(self.data.length_val / self.batch_size)),
            "test": int(np.ceil(self.data.length_test / self.batch_size))
        }
        
        self.cur_idx  = 0
        
    def load_batch(self, split):
        for batch in range(self.num_batches[split]):
            yield self.data.images_train[self.cur_idx:self.cur_idx+self.batch_size], self.data.labels_train[self.cur_idx:self.cur_idx+self.batch_size]
            self.cur_idx = self.cur_idx + self.batch_size