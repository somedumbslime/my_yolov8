import cv2
import numpy as np
from blocks import Conv2d, BatchNorm2d, SiLU, MSE, CBS, SGD



path = r"C:\Users\sanek\OneDrive\Рабочий стол\projects\rt_yolo_game\data\video_2025-05-27_13-49-46_frame_003.jpg"
file_bytes = np.fromfile(path, dtype=np.uint8)
test_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
test_image = np.expand_dims(test_image, axis=0)
test_image = np.transpose(test_image, (0, 3, 1, 2))

# LABEL mask for this photo
x_min, y_min = 305, 300
x_max, y_max = 336, 350

gt_mask = np.zeros((640, 640))
gt_mask[y_min:y_max, x_min:x_max] = 1

target_h,  target_w = 80, 80
gt_mask_resized = cv2.resize(gt_mask, (target_h, target_w), interpolation=cv2.INTER_NEAREST)
gt_label = gt_mask_resized[np.newaxis, np.newaxis, ...]
gt_label = np.broadcast_to(gt_label, (1, 256, target_h, target_w))

# ----

# other ways to do dummy images
# test_image = np.random.randint(0, 255, size=(3, 640, 640))
# test_image = np.transpose(np.stack((np.zeros((640, 640)), np.zeros((640, 640)), np.zeros((640, 640))), axis=-1), (2, 0, 1))


    
conv1 = Conv2d(c_in=3, c_out=64, k=3, s=2, p=1)
conv2 = Conv2d(c_in=64, c_out=128, k=3, s=2, p=1)
conv3 = Conv2d(c_in=128, c_out=256, k=3, s=2, p=1)

bn1 = BatchNorm2d(64, training=True)
bn2 = BatchNorm2d(128, training=True)
bn3 = BatchNorm2d(256, training=True)

act1  = SiLU()
act2  = SiLU()
act3  = SiLU()

mse = MSE()

optim = SGD()

output1 = conv1.forward(test_image) # conv ->  batchnorm -> act (SiLU) | currently only "conv"
output_norm1 = bn1.forward(output1)
output_act1 = act1.forward(output_norm1)

output2 = conv2.forward(output_act1)
output_norm2 = bn2.forward(output2)
output_act2 = act2.forward(output_norm2)

output3 = conv3.forward(output_act2)
output_norm3 = bn3.forward(output3)
output_act3 = act3.forward(output_norm3)

# print(np.round(output_act3, 2))
output_mse_f = mse.forward(output_act3, gt_label)
# print("output_mse_f:", output_mse_f)

output_mse_b = mse.backward()
# print("output_mse_b:", output_mse_b.shape)

output_act3_b = act3.backward(output_mse_b)
# print("output_act3_b:", output_act3.shape)

output_norm3_b = bn3.backward(output_act3_b)
# print("output_norm3_b:", output_norm3_b.shape)

output_conv3_b = conv3.backward(output_norm3_b)
# print("output_conv3_b:", output_conv3_b.shape)

optim.step()


# output_act1 = output_act1
# output_act2 = output_act2
# output_act3 = output_act3

# output_concat0 = np.hstack((output_act1[0][0], output_act1[0][1])) # P1
output_concat1 = np.hstack((output_act2[0][0], output_act2[0][1], output_act2[0][2], output_act2[0][3])) # P2
output_concat2 = np.hstack((output_act2[0][4], output_act2[0][5], output_act2[0][6], output_act2[0][7])) # P2

output_concat3 = np.hstack((output_act3[0][0], output_act3[0][1], output_act3[0][2], output_act3[0][3], output_act3[0][4], output_act3[0][5], output_act3[0][6], output_act3[0][7])) # P3
output_concat4 = np.hstack((output_act3[0][8], output_act3[0][9], output_act3[0][10], output_act3[0][11], output_act3[0][12], output_act3[0][13], output_act3[0][14], output_act3[0][15])) # P3
output_concat = np.vstack((output_concat1, output_concat2, output_concat3, output_concat4))

# cv2.imshow("win", np.transpose(output_norm, (1, 2, 0)).astype(np.uint8))
output_concat = cv2.resize(output_concat, None, fx=2, fy=2)
cv2.imshow("win", output_concat)

if cv2.waitKey(0) == ord("q"):
    cv2.destroyAllWindows()