import cv2
import numpy as np
from blocks import Conv2d, BatchNorm2d, SiLU, MSE, CBS, SGD
from data import Dataset, Dataloader

data = Dataset(r"C:\Users\sanek\OneDrive\Рабочий стол\projects\rt_yolo_game\data\learning_mini_1000_v1\learning_mini_1000_v1")
dataloader = Dataloader(data, batch_size=8)

conv_block = CBS(3, 4)
optim = SGD(lr=0.01)
loss_func = MSE()

for ep in range(100):
    train_gen = dataloader.load_batch("train")
    
    for b in range(dataloader.num_batches["train"]):
        images, labels = next(train_gen)
        # print(f"batch={b}, shape= {images.shape}")
        optim.zero_grad(conv_block)
        pred = conv_block.forward(images)
        # target = 
        loss = loss_func.forward(pred, target)
        grad = loss_func.backward()
        conv_block.backward(grad)
        print(
        f"Ep={ep} | "
        f"Loss={loss:.6f} | "
        # f"dW norm={np.linalg.norm(conv_block.conv.dW):.6e} | "
        # f"dW absmax={np.max(np.abs(conv_block.conv.dW)):.6e} | "
        # f"dgamma norm={np.linalg.norm(conv_block.bn.dgamma):.6e} | "
        # f"dgamma absmax={np.max(np.abs(conv_block.bn.dgamma)):.6e}"
        f"dW norm={np.linalg.norm(conv_block.dW):.6e} | "
        f"dW absmax={np.max(np.abs(conv_block.dW)):.6e} | "
    )
        
        optim.step(conv_block)
    