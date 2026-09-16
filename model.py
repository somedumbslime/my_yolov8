from blocks import Conv2d, BatchNorm2d, SiLU, MSE


class Model:
    def __init__(self, c_in, c_out, training=False):
        self.conv1 = Conv2d(c_in, 64)
        self.bn1 = BatchNorm2d(64, training)
        self.act1 = SiLU()
        
        self.conv2 = Conv2d(64, 128)
        self.bn2 = BatchNorm2d(128, training)
        self.act2 = SiLU()
        
        self.conv3 = Conv2d(128, 256)
        self.bn3 = BatchNorm2d(256, training)
        self.act3 = SiLU()
        
        self.mse = MSE()
        
    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        
        return x