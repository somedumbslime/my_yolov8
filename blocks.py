import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import cv2



class Conv2d:
    def __init__(self, c_in, c_out, k=3, s=2, p=1):
        self.k = k
        self.s = s
        self.p = p
        self.c_in = c_in
        self.c_out = c_out
        
        self.W = np.random.randn(c_out, c_in, k, k) * 0.01
        self.b = np.ones((c_out))
        
        self.x = None
        self.x_padded = None
        self.X_col = None
        self.W_col = None
        
        self.db = None
        self.dW = None
    
    
    def forward(self, x):
        if self.p > 0:
            x_padded = np.pad(x, pad_width=((0, 0), (0, 0), (self.p, self.p), (self.p, self.p)), mode="constant", constant_values=0)
        else:
            x_padded = x
        
        batch, c_in, h_in, w_in = x_padded.shape
        
        windows = sliding_window_view(
            x_padded,
            (self.k, self.k),
            axis=(2, 3)
            )
        
        windows = windows[:, :, ::self.s, ::self.s, :, :]
        # print("windows:", windows.shape)
        windows = np.transpose(windows, (0, 2, 3, 1, 4, 5))
        
        c_out = self.W.shape[0]
        h_out = ((h_in - self.k) // self.s) + 1
        w_out = ((w_in - self.k) // self.s) + 1
        
        X_col = windows.reshape(batch, h_out * w_out, c_in * self.k * self.k)
        W_col = self.W.reshape(c_out, c_in * self.k * self.k)
        
        self.x = x
        self.x_padded = x_padded
        self.X_col = X_col
        self.W_col = W_col
        
        out_col = np.dot(X_col, W_col.T)
        out_col += self.b

        out = out_col.reshape(batch, h_out, w_out, c_out)
        out = np.transpose(out, (0, 3, 1, 2))

        return out
    
    def backward(self, dout):
        # dout shape from (N,Cout,Hout,Wout) to (N,Hout*Wout,Cout) , from bn.backward=(1, 256, 80, 80) to out_col.forward=(1, 6400, 256)
        # in P3 layer: X_col=(1, 6400, 1152), W_col=(256, 1152), b=(1, 256, 1, 1)
        N, C, H, W = dout.shape
        _, _, K = self.X_col.shape
        dout_col = np.transpose(dout, (0, 2, 3, 1))
        dout_col = dout_col.reshape((N, H * W, C))
        
        # print("x:", self.x.shape)
        # print("dout:", dout.dtype)
        # print("dout_col:", dout_col.shape)
        # print("X_col:", self.X_col.shape)
        # print("W_col:", self.W_col.shape)
        # print()
        
        db = np.sum(dout, axis=(0, 2, 3))
        # print("db:", db.shape)
        
        dout_2d = dout_col.reshape(-1, C)
        X_col_2d = self.X_col.reshape(-1, K)
        dW = np.dot(dout_2d.T, X_col_2d).reshape(self.c_out, self.c_in, self.k, self.k)
        # dW = (dW / dout_2d.shape[0]).reshape(self.c_out, self.c_in, self.k, self.k)
        # print("dw:", dw.shape)
        
        dx_col = np.dot(dout_col, self.W_col)
        # print("dx_col:", dx_col.shape)
        
        dx_windows = dx_col.reshape(N, H, W, self.c_in, self.k, self.k)
        dx_padded = np.zeros_like(self.x_padded, dtype=np.float64)
        # print("dx_paded: ", dx_padded.dtype) 
        for ki in range(self.k):
            for kj in range(self.k):
                dx_padded[
                    :,
                    :,
                    ki: ki + H * self.s : self.s,
                    kj: kj + W * self.s : self.s
                ] += dx_windows[:, :, :, :, ki, kj].transpose(0, 3, 1, 2)
        
        if self.p > 0:
            dx = dx_padded[:, :, self.p:-self.p, self.p:-self.p]
        else:
            dx = dx_padded
        
        self.db = db
        self.dW = dW
        
        # print("dx:", dx.shape)
        return dx


class BatchNorm2d:
    def __init__(self, num_features=64, training=True, eps=0.00001):
        self.gamma = np.ones((1, num_features, 1, 1))
        self.beta = np.zeros((1, num_features, 1, 1))
        
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))
        self.momentum = 0.1
        
        self.training = training
        self.eps = eps
    
        self.x_norm = None
        self.mean = None
        self.var = None
        self.x_mu = None
        self.dgamma = None
        self.dbeta = None
    
    
    def forward(self, x):
        if self.training ==  True:
            mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            var = np.mean((x - mean)**2, axis=(0, 2, 3), keepdims=True)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var 
        else:
            mean = self.running_mean
            var = self.running_var
        
        x_norm = (x - mean) / (np.sqrt(var + self.eps))
        self.x_norm = x_norm
        self.mean = mean
        self.var = var
        self.x_mu = x - mean
        
        y = self.gamma * x_norm + self.beta
        
        return y

    def backward(self, dout):
        N, C, H, W = dout.shape
        M = N * H * W
        dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
        dgamma = np.sum(dout * self.x_norm, axis=(0, 2, 3), keepdims=True)
        self.dbeta = dbeta
        self.dgamma = dgamma
        
        dx_norm = dout * self.gamma
        
        invstd = 1 / (np.sqrt(self.var + self.eps))

        dx_mu_1 = dx_norm * invstd
        dinvstd = np.sum(dx_norm * self.x_mu, axis=(0, 2, 3), keepdims=True)
        dvar = dinvstd * (-1/2) *  (self.var + self.eps)**(-3/2)
        dx_mu_2 = dvar * (2 * self.x_mu / M)
        
        dx_mu = dx_mu_1 + dx_mu_2

        dx_1 = dx_mu
        dmu = -np.sum(dx_mu, axis=(0, 2, 3), keepdims=True)
        dx_2 = dmu / M

        dx = dx_1 + dx_2
        # print("bn dx:", dx.shape)
        return dx

class SiLU:  
    def __init__(self):
        self.x = None
        
    def forward(self, x):
        # x_clipped = np.clip(x, -50, 50)
        self.x = x
        sigm = 1 / (1 + np.exp(-x))
        y = x * sigm

        return y
    
    def backward(self, dout):
        sigm = 1 / (1 + np.exp(-self.x))
        local_derivative = sigm + self.x * sigm * (1 - sigm)
        dx = dout * local_derivative
        
        return dx


class MSE:
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, y, t):
        self.y, self.t = y, t
        loss = np.mean((y - t)**2)
    
        return loss
    
    def backward(self):
        N = self.y.size
        dx = (2 / N) * (self.y - self.t)

        return dx

    

class CBS:
    def __init__(self, c_in, c_out, k=3, s=2, p=1):
        self.conv = Conv2d(c_in, c_out, k, s, p)
        self.bn = BatchNorm2d(c_out)
        self.act = SiLU()
        self.y = None
        
    def forward(self, x):
        y = self.act.forward(self.bn.forward(self.conv.forward(x)))
        self.y = y
        
        return y
    
    def backward(self, grad_loss):
        dx_silu = self.act.backward(grad_loss)
        dx_bn = self.bn.backward(dx_silu)
        dx_conv = self.conv.backward(dx_bn)
        
        return None
    
    
class SGD:
    def __init__(self, lr=0.001):
        self.lr = lr
    
    def step(self, block):
        if type(block) == (CBS):
        # BatchNorm2d
            block.bn.gamma = block.bn.gamma - block.bn.dgamma * self.lr
            block.bn.beta = block.bn.beta - block.bn.dbeta * self.lr
            # Conv
            block.conv.b = block.conv.b - block.conv.db * self.lr
            block.conv.W = block.conv.W - block.conv.dW * self.lr
        
        if type(block) == (Conv2d):
            block.b = block.b - block.db * self.lr
            block.W = block.W - block.dW * self.lr
        return None
        
    def zero_grad(self, block):
        if type(block) == (CBS):
            # BatchNorm2d
            block.bn.dgamma = np.zeros_like(block.bn.dgamma)
            block.bn.dbeta = np.zeros_like(block.bn.dbeta)
            # Conv
            block.conv.db = np.zeros_like(block.conv.db)
            block.conv.dW = np.zeros_like(block.conv.dW)
            
        if type(block) == (Conv2d):
            block.db = np.zeros_like(block.db)
            block.dW = np.zeros_like(block.dW)