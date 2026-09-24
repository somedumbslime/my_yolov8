import cv2
import numpy as np
from blocks import Conv2d, BatchNorm2d, SiLU, MSE, CBS, SGD
from data import Dataset, Dataloader

data = Dataset(r"C:\Users\sanek\OneDrive\Рабочий стол\projects\rt_yolo_game\data\learning_mini_1000_v1\learning_mini_1000_v1")
dataloader = Dataloader(data, batch_size=8)

# print(data.labels_train[0])
# exit()

conv_block = CBS(3, 4)
optim = SGD(lr=0.01)
loss_func = MSE()

for ep in range(100):
    train_gen = dataloader.load_batch("train")
    
    for b in range(dataloader.num_batches["train"]):
        images, labels = next(train_gen)
        # img0 = images[0]
        img0 = np.transpose(images[4], (1, 2, 0))
        label0 = np.array(np.transpose(np.multiply(labels[4][1], 255), (1, 2, 0)), dtype=np.uint8)
        label0_3ch = np.array(cv2.cvtColor(label0, cv2.COLOR_GRAY2BGR))
        # print(img0.shape)
        # print(label0_3ch.shape)
        out_photo = np.concatenate((img0, label0_3ch), 1)
        cv2.imshow("win", out_photo)
        if cv2.waitKey(0) == ord("q"):
            cv2.destroyAllWindows()
        # print(labels)

        # print(f"batch={b}, shape= {images.shape}")
        optim.zero_grad(conv_block)
        pred = conv_block.forward(images)
        print(pred.shape)
        
        exit()
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
    