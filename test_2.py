import cv2
import numpy as np
from blocks import Conv2d, BatchNorm2d, SiLU, MSE, CBS, SGD



path = r"C:\Users\sanek\OneDrive\Рабочий стол\projects\rt_yolo_game\data\video_2025-05-27_13-49-46_frame_003.jpg"
file_bytes = np.fromfile(path, dtype=np.uint8)
test_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
test_image = np.expand_dims(test_image, axis=0)
test_image = np.transpose(test_image, (0, 3, 1, 2))

# LABEL mask for this photo
# x_min, y_min = 305, 300
# x_max, y_max = 336, 350

# gt_mask = np.zeros((640, 640), dtype=np.uint8)
# gt_mask[y_min:y_max, x_min:x_max] = 1

# target_h,  target_w = 320, 320
# gt_mask_resized = cv2.resize(gt_mask, (target_h, target_w), interpolation=cv2.INTER_NEAREST)
# gt_label = gt_mask_resized[np.newaxis, np.newaxis, ...]
# gt_label = np.broadcast_to(gt_label, (1, 16, target_h, target_w))
# print("gt_label:", gt_label.dtype)

# DUMMY IMAGE
dummy_img = np.random.random((2, 3, 8, 8))

conv_block = Conv2d(3, 4)

# conv_block = CBS(3, 4)
optim = SGD(lr=0.01)
loss_func = MSE()
target = np.random.random((2, 4, 4, 4))

for ep in range(100):
    optim.zero_grad(conv_block)
    pred = conv_block.forward(dummy_img)
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
    