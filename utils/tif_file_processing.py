import numpy as np
from osgeo import gdal

# 读取tif
def read_tif(path):
    dataset = gdal.Open(path)
    cols = dataset.RasterXSize # 图像长度
    rows = dataset.RasterYSize # 图像宽度
    im_proj = (dataset.GetProjection()) # 读取投影
    im_Geotrans = (dataset.GetGeoTransform()) # 读取仿射变换
    im_data = dataset.ReadAsArray(0, 0, cols, rows) # 转为numpy格式
    if len(im_data.shape)<3:
        im_data = np.expand_dims(im_data, 0)
    im_data = np.transpose(im_data, [2, 1, 0])
    del dataset
    return im_data, im_Geotrans, im_proj, cols, rows

# 写出tif
def write_tif(newpath, im_data, im_geotrans, im_proj, datatype):
    # datatype常用gdal.GDT_UInt16 gdal.GDT_Int16 gdal.GDT_Float32
    if len(im_data.shape)==3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
    driver = gdal.GetDriverByName('GTiff')
    new_dataset = driver.Create(newpath, im_width, im_height, im_bands, datatype)
    new_dataset.SetGeoTransform(im_geotrans)
    new_dataset.SetProjection(im_proj)

    if im_bands == 1:
        new_dataset.GetRasterBand(1).WriteArray(im_data.reshape(im_height, im_width))
    else:
        for i in range(im_bands):
            new_dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del new_dataset