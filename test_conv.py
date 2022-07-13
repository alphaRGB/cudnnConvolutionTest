import sys
import numpy as np
from build import libconv2d
import cv2
print(libconv2d)


def image_conv2d():
    image = cv2.imread('/home/penghuiwei/MyWorkspace/ubuntu/CPP/cudaCudnnConv/image/lena.jpg')

    def make_kernel_3x3(in_channels, out_channels):
        data = []
        kernel_tmp = [1,1,1,1,-8,1,1,1,1]
        for j in range(out_channels):
            for i in range(in_channels):
                data.append(kernel_tmp)
        return np.array(data, dtype=np.float32).reshape([out_channels, in_channels, 3, 3])
        
    x_gpu = libconv2d.Tensor()
    w_gpu = libconv2d.Tensor()
    x_gpu.from_numpy(image.astype(np.float32).reshape(1, 3, 512, 512))
    w_gpu.from_numpy(make_kernel_3x3(3, 3))
    
    param = libconv2d.Conv2dParam()
    param.pad_h = 1
    param.pad_w = 1
    param.dilation_h = 1
    param.dilation_w = 1
    param.u = 1
    param.v = 1

    y_gpu = libconv2d.Tensor()
    libconv2d.cudnn_conv2d_out(input_gpu=x_gpu, weight_gpu=w_gpu, params=param, output_gpu=y_gpu)
    
    dst = y_gpu.numpy()

    dst = dst.astype(np.int32).squeeze(0)    
    dst = dst.clip(0, 255)
    dst = cv2.convertScaleAbs(dst)
    # dst = np.transpose(dst, [1, 2, 0])
    # dst = dst.reshape([512, 512, 3])
    cv2.imwrite('conv.png', dst)


if __name__ == '__main__':
    image_conv2d()