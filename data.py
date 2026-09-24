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
        self.labels_train = self.unpack_labels(self.labels_train)
        
    def unpack_labels(self, labels_list):
        labels = []
        
        for label in labels_list:
            with open(label, "r", encoding="utf-8") as file:
                line = [x.strip("\n").split(" ", maxsplit=1) for x in file.readlines()]
                for i in range(len(line)):
                    a, b = line[i]
                    a = int(a)
                    b = [float(x) for x in b.split()]
                    line[i] = [a, b]
                    # print(line[i])
                    
                             
                # print(line)
                labels.append(line)
        
        return labels
    
    
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
        
    @staticmethod
    def load_labels(labels):
        labels_ready = []
        for b in range(len(labels)):
            classes = []
            boxes = []
            box_arr = np.zeros((640, 640))
            for label in labels[b]:
                cls, box = label
                x, y, w, h = [min(round(x*640), 640) for x in box]
                x = x - w // 2
                y = y - h // 2
                box_arr[y: y+h, x:x+w] = 1
                classes.append(cls)
            boxes.append(box_arr)
                
            labels_ready.append([classes, boxes])
        
        return labels_ready

    def load_batch(self, split):
        for batch in range(self.num_batches[split]):
            # костиль
            images = self.data.images_train[self.cur_idx:self.cur_idx+self.batch_size]
            images = np.transpose(np.array(images), (0, 3, 1, 2))
            labels = self.data.labels_train[self.cur_idx:self.cur_idx+self.batch_size]
            labels = self.load_labels(labels)
                
            yield images, labels
            self.cur_idx = self.cur_idx + self.batch_size