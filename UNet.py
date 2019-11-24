import torch
import torch.nn as nn

class UNet(nn.Module):
  
  def __init__(self):
    super(UNet, self).__init__()

    self.conv1 = nn.Conv2d(60, 1920, kernel_size=3, stride=2, bias=False)
    self.conv2 = nn.Conv2d(1920, 1920, kernel_size=3, stride=2, bias=False)
    self.norm1 = nn.BatchNorm2d(1920)
    self.max_pool1 = nn.MaxPool2d(kernel_size=2)

    self.conv3 = nn.Conv2d(1920, 3840, kernel_size=3, stride=2, bias=False)
    self.conv4 = nn.Conv2d(3840, 3840, kernel_size=3, stride=2, bias=False)
    self.norm2 = nn.BatchNorm2d(3840)
    self.max_pool2 = nn.MaxPool2d(kernel_size=2)

    self.conv5 = nn.Conv2d(3840, 7680, kernel_size=3, stride=2, bias=False)
    self.conv6 = nn.Conv2d(7680, 7680, kernel_size=3, stride=2, bias=False)
    self.norm3 = nn.BatchNorm2d(7680)
    self.max_pool3 = nn.MaxPool2d(kernel_size=2)

    self.conv7 = nn.Conv2d(7680, 15360, kernel_size=3, stride=2, bias=False)
    self.conv8 = nn.Conv2d(15360, 15360, kernel_size=3, stride=2, bias=False)
    self.norm3 = nn.BatchNorm2d(15360)
    self.max_pool4 = nn.MaxPool2d(kernel_size=2)

    self.conv9 = nn.Conv2d(15360, 30720, kernel_size=3, stride=2, bias=False)
    self.conv10 = nn.Conv2d(30720, 30720, kernel_size=3, stride=2, bias=False)
    self.norm = nn.BatchNorm2d(15360)

    self.upconv4 = nn.ConvTranspose2d(30720, 15360, kernel_size=2, stride=2)
    self.conv11 = nn.Conv2d(15360, 15360, kernel_size=3, stride=2, bias=False)
    self.conv12 = nn.Conv2d(15360, 15360, kernel_size=3, stride=2, bias=False)
    self.norm5 = nn.BatchNorm2d(15360)

    self.upconv3 = nn.ConvTranspose2d(15360, 7680, kernel_size=2, stride=2)
    self.conv13 = nn.Conv2d(7680, 7680, kernel_size=3, stride=2, bias=False)
    self.conv14 = nn.Conv2d(7680, 7680, kernel_size=3, stride=2, bias=False)
    self.norm6 = nn.BatchNorm2d(15360)

    self.upconv2 = nn.ConvTranspose2d(7680, 3840, kernel_size=2, stride=2)
    self.conv15 = nn.Conv2d(3840, 3840, kernel_size=3, stride=2, bias=False)
    self.conv16 = nn.Conv2d(3840, 3840, kernel_size=3, stride=2, bias=False)
    self.norm7 = nn.BatchNorm2d(15360)

    self.upconv1 = nn.ConvTranspose2d(3840, 1920, kernel_size=2, stride=2)
    self.conv17 = nn.Conv2d(1920, 1920, kernel_size=3, stride=2, bias=False)
    self.conv18 = nn.Conv2d(1920, 1920, kernel_size=3, stride=2, bias=False)
    self.norm8 = nn.BatchNorm2d(15360)

    self.conv = nn.Conv2d(1920, 60, kernel_size=3, stride=2)

  def foward(self, x):

    enc1 = nn.ReLu(self.norm1(self.conv1(x)))
    enc1 = nn.ReLu(self.max_pool1(self.norm1(self.conv2(enc1))))

    enc2 = nn.ReLu(self.norm2(self.conv3(enc1)))
    enc2 = nn.ReLu(self.max_pool2(self.norm2(self.conv4(enc2))))

    enc3 = nn.ReLu(self.norm3(self.conv5(enc2)))
    enc3 = nn.ReLu(self.max_pool3(self.norm3(self.conv6(enc3))))

    enc4 = nn.ReLu(self.norm4(self.conv7(enc3)))
    enc4 = nn.ReLu(self.max_pool4(self.norm4(self.conv8(enc4))))

    bn = self.ReLu(self.norm(self.conv9(enc4)))
    bn = self.ReLu(self.norm(self.conv10(bn)))

    dec4 = self.upconv4(bn)
    dec4 = torch.cat((dec4, enc4), dim=1)
    dec4 = nn.ReLu(self.conv11(dec4))
    dec4 = nn.ReLu(self.norm5(self.conv12(dec4)))

    dec3 = self.upconv4(dec4)
    dec3 = torch.cat((dec3, enc3), dim=1)
    dec3 = nn.ReLu(self.conv13(dec3))
    dec3 = nn.ReLu(self.norm6(self.conv14(dec3)))

    dec2 = self.upconv4(dec3)
    dec2 = torch.cat((dec2, enc2), dim=1)
    dec2 = nn.ReLu(self.conv15(dec2))
    dec2 = nn.ReLu(self.norm7(self.conv16(dec2)))

    dec1 = self.upconv4(dec2)
    dec1 = torch.cat((dec1, enc1), dim=1)
    dec1 = nn.ReLu(self.conv17(dec1))
    dec1 = nn.ReLu(self.norm8(self.conv18(dec1)))

    return torch.sigmoid(self.conv(dec1))