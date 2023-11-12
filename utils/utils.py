import numpy as np
import torchvision.transforms as transforms

# a class for calculating the average of the accuracy and the loss
#-------------------------------------------------------------------------------
class AverageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.average = 0 
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.count += n
    self.average = self.sum / self.count
#-------------------------------------------------------------------------------

def transform(IsResize,Resize_size,IsTotensor,IsNormalize,Norm_mean,Norm_std,IsRandomGrayscale,IsColorJitter,
              brightness,contrast,hue,saturation,IsCentercrop,Centercrop_size,IsRandomCrop,RandomCrop_size,
              IsRandomResizedCrop,RandomResizedCrop_size,Grayscale_rate,IsRandomHorizontalFlip,HorizontalFlip_rate,
              IsRandomVerticalFlip,VerticalFlip_rate,IsRandomRotation,degrees):

  transform_list = []

    #-----------------------------------------------<旋转图像>-----------------------------------------------------------#
  if IsRandomRotation:
    transform_list.append(transforms.RandomRotation(degrees))
  if IsRandomHorizontalFlip:
    transform_list.append(transforms.RandomHorizontalFlip(HorizontalFlip_rate))
  if IsRandomVerticalFlip:
    transform_list.append(transforms.RandomHorizontalFlip(VerticalFlip_rate))

    #-----------------------------------------------<图像颜色>-----------------------------------------------------------#
  if IsColorJitter:
    transform_list.append(transforms.ColorJitter(brightness,contrast,saturation,hue))
  if IsRandomGrayscale:
    transform_list.append(transforms.RandomGrayscale(Grayscale_rate))

    #---------------------------------------------<缩放或者裁剪>----------------------------------------------------------#
  if IsResize:
    transform_list.append(transforms.Resize(Resize_size))
  if IsCentercrop:
    transform_list.append(transforms.CenterCrop(Centercrop_size))
  if IsRandomCrop:
    transform_list.append(transforms.RandomCrop(RandomCrop_size))
  if IsRandomResizedCrop:
    transform_list.append(transforms.RandomResizedCrop(RandomResizedCrop_size))

    #---------------------------------------------<tensor化和归一化>------------------------------------------------------#
  if IsTotensor:
    transform_list.append(transforms.ToTensor())
  if IsNormalize:
    transform_list.append(transforms.Normalize(Norm_mean,Norm_std))

    # 您可以更改数据增强的顺序，但是数据增强的顺序可能会影响最终数据的质量，因此除非您十分明白您在做什么,否则,请保持默认顺序
  # transforms_order=[Resize_transform,Rotation,Color,Tensor,Normalize]
  return transforms.Compose(transform_list)


def get_transform(size=[200, 200], mean=[0, 0, 0], std=[1, 1, 1], IsResize=False, IsCentercrop=False, IsRandomCrop=False, IsRandomResizedCrop=False, IsTotensor=False, IsNormalize=False, IsRandomGrayscale=False, IsColorJitter=False, IsRandomVerticalFlip=False, IsRandomHorizontalFlip=False, IsRandomRotation=False):
  diy_transform = transform(
      IsResize=IsResize, #是否缩放图像
      Resize_size=size, #缩放后的图像大小 如（512,512）->（256,192）
      IsCentercrop=IsCentercrop,#是否进行中心裁剪
      Centercrop_size=size,#中心裁剪后的图像大小
      IsRandomCrop=IsRandomCrop,#是否进行随机裁剪
      RandomCrop_size=size,#随机裁剪后的图像大小
      IsRandomResizedCrop=IsRandomResizedCrop,#是否随机区域进行裁剪
      RandomResizedCrop_size=size,#随机裁剪后的图像大小
      IsTotensor=IsTotensor, #是否将PIL和numpy格式的图片的数值范围从[0,255]->[0,1],且将图像形状从[H,W,C]->[C,H,W]
      IsNormalize=IsNormalize, #是否对图像进行归一化操作,即使用图像的均值和方差将图像的数值范围从[0,1]->[-1,1]
      Norm_mean=mean,#图像的均值，用于图像归一化，建议使用自己通过计算得到的图像的均值
      Norm_std=std,#图像的方差，用于图像归一化，建议使用自己通过计算得到的图像的方差
      IsRandomGrayscale=IsRandomGrayscale,#是否随机将彩色图像转化为灰度图像
      Grayscale_rate=0.5,#每张图像变成灰度图像的概率，设置为1的话等同于transforms.Grayscale()
      IsColorJitter=IsColorJitter,#是否随机改变图像的亮度、对比度、色调和饱和度
      brightness=0.5,#每个图像被随机改变亮度的概率
      contrast=0.5,#每个图像被随机改变对比度的概率
      hue=0.5,#每个图像被随机改变色调的概率
      saturation=0.5,#每个图像被随机改变饱和度的概率
      IsRandomVerticalFlip=IsRandomVerticalFlip,#是否垂直翻转图像
      VerticalFlip_rate=0.5,#每个图像被垂直翻转图像的概率
      IsRandomHorizontalFlip=IsRandomHorizontalFlip,#是否水平翻转图像
      HorizontalFlip_rate=0.5,#每个图像被水平翻转图像的概率
      IsRandomRotation=IsRandomRotation,#是是随机旋转图像
      degrees=10,#每个图像被旋转角度的范围 如degrees=10 则图像将随机旋转一个(-10,10)之间的角度
  )
  return diy_transform