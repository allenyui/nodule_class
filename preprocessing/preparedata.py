import SimpleITK as sitk
import numpy as np
import os
from glob import glob
import pandas as pd
from tqdm import tqdm


def load_itkfilewithtrucation(filename, upper=200, lower=-200):
    """
    load mhd files,set truncted value range and normalization 0-255
    :param filename:
    :param upper:
    :param lower:
    :return:
    """
    # 1,tructed outside of lung value
    srcitkimage = sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32)
    srcitkimagearray = sitk.GetArrayFromImage(srcitkimage)
    # 截断 目的是去除图像中的极值或噪声
    srcitkimagearray[srcitkimagearray > upper] = upper
    srcitkimagearray[srcitkimagearray < lower] = lower
    # 2,get tructed outside of lung value image
    sitktructedimage = sitk.GetImageFromArray(srcitkimagearray)
    origin = np.array(srcitkimage.GetOrigin())
    spacing = np.array(srcitkimage.GetSpacing())
    sitktructedimage.SetSpacing(spacing)
    sitktructedimage.SetOrigin(origin)
    # 3 normalization value to 0-255
    rescalFilt = sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(255)
    rescalFilt.SetOutputMinimum(0)
    itkimage = rescalFilt.Execute(sitk.Cast(sitktructedimage, sitk.sitkFloat32))
    return itkimage


# Some helper functions

def get_cube_from_img(img3d, center, block_size):
    # get roi(z,y,z) image and in order the out of img3d(z,y,x)range
    center_z = center[0]
    center_y = center[1]
    center_x = center[2]
    # 确定提取的立方体（3D 区域）在 3D 图像中的起始坐标
    start_x = max(center_x - block_size / 2, 0)
    if start_x + block_size > img3d.shape[2]:
        start_x = img3d.shape[2] - block_size
    start_y = max(center_y - block_size / 2, 0)
    if start_y + block_size > img3d.shape[1]:
        start_y = img3d.shape[1] - block_size
    start_z = max(center_z - block_size / 2, 0)
    if start_z + block_size > img3d.shape[0]:
        start_z = img3d.shape[0] - block_size
    start_z = int(start_z)
    start_y = int(start_y)
    start_x = int(start_x)
    roi_img3d = img3d[start_z:start_z + block_size, start_y:start_y + block_size, start_x:start_x + block_size]
    return roi_img3d


# Helper function to get rows in data frame associated
# with each file
def get_filename(file_list, case):
    for f in file_list:
        if case in f.lower():
            return f
        # print(f"Warning: Could not find file with case {case}")
        return None


def get_node_classify():
    # Getting list of image files and output nuddle 0 and 1
    for subsetindex in range(10):
        classify_size = 48
        luna_path = r"E:\Workplace\dataset"
        luna_subset_path = os.path.join(luna_path, "subset" + str(subsetindex))
        print(f"正在查找的目录：{luna_subset_path}")
        output_path = r"E:\Workplace\dataset\classification"
        file_list = glob(os.path.join(luna_subset_path, "*.mhd"))
        # print(f"正在查找的文件：{file_list}")

        # file_list_path = []
        # for i in range(len(file_list)):
        # file_list_path.append(file_list[i][0:-4])  # 去掉文件扩展名'.mhd' 以匹配csv中的文件名
        # print(file_list_path[:10])

        # The locations of the nodes
        luna_csv_path = r"E:\Workplace\dataset\luna16"
        df_node = pd.read_csv(luna_csv_path + "/CSVFILES/" + "candidates.csv")
        # print(df_node["seriesuid"].unique()[:10])
        # print(df_node.head())
        test_seriesuid = df_node["seriesuid"].iloc[0]
        print(get_filename(file_list, test_seriesuid))
        # 找到每个结节的对应文件
        df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name.lower()))
        df_node = df_node.dropna()  # 移除NaN值的行
        # Looping over the image files
        for fcount, img_file in enumerate(tqdm(file_list)):
            # print(f"Processing file: {img_file}")
            # get all nodules associate with file
            mini_df = df_node[df_node["file"] == img_file]
            # print(f"Number of nodes in file: {len(mini_df)}")
            # some files may not have a nodule--skipping those
            if mini_df.shape[0] > 0:
                # img_file = img_file+".mhd"
                # load the data once
                itk_img = load_itkfilewithtrucation(img_file, 600, -1000)
                img_array = sitk.GetArrayFromImage(itk_img)
                # x,y,z  Origin in world coordinates (mm)
                origin = np.array(itk_img.GetOrigin())
                # spacing of voxels in world coor. (mm)
                spacing = np.array(itk_img.GetSpacing())
                # go through all nodes
                index = 0
                # 遍历‘mini_df’ 中所有结节
                for node_idx, cur_row in mini_df.iterrows():
                    # 获取结节x，y，z坐标
                    node_x = cur_row["coordX"]
                    node_y = cur_row["coordY"]
                    node_z = cur_row["coordZ"]
                    label = cur_row["class"]  # 获取结节标签
                    # nodule center
                    center = np.array([node_x, node_y, node_z])
                    # nodule center in voxel space (still x,y,z ordering)  # clip prevents going out of bounds in Z
                    v_center = np.rint((center - origin) / spacing)
                    # convert x,y,z order v_center to z,y,z order v_center
                    v_center[0], v_center[1], v_center[2] = v_center[2], v_center[1], v_center[0]
                    # get cub size of classify_size
                    node_cube = get_cube_from_img(img_array, v_center, classify_size)
                    node_cube.astype(np.uint8)
                    # save as npy file
                    if label == 1:
                        filepath = output_path + "1/"
                        if not os.path.exists(filepath):
                            os.makedirs(filepath)
                        filename = str(subsetindex) + "_" + str(fcount) + "_" + str(index)
                        np.save(filepath + filename + ".npy", node_cube)
                        print("File saved: ", filepath + filename + ".npy")
                    if label == 0:
                        filepath = output_path + "0/"
                        if not os.path.exists(filepath):
                            os.makedirs(filepath)
                        filename = str(subsetindex) + "_" + str(fcount) + "_" + str(index)
                        np.save(filepath + filename + ".npy", node_cube)
                        print("File saved: ", filepath + filename + ".npy")
                    index += 1  # 更新已经处理的结节的数量，并确保每个结节有一个独一无二的文件名


get_node_classify()  # 开始执行该函数
