from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models
import torchsummary
from torchsummary import summary
import csv
import os

from models import densenet_3d

file_dir = r"E:\Workplace\dataset\classification_aug1/"  # npy文件路径
# file_dir = r"E:\Workplace\dataset\classification0/"  # npy文件路径
save_dir = r"../data/featuremapping/feature_maps_dense_355_36_1"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device =',device)
print(torch.cuda.get_device_name(0))

def generate_model(model_type='densenet_3d', model_depth=121,bn_size=4,
                     drop_rate=0,
                     num_classes=4):
    assert model_type in [
        'densenet_3d'
    ]

    if model_type == 'densenet_3d':
        assert model_depth in [121, 169, 201, 264]

    if model_depth == 121:
        model = densenet_3d.densenet121(bn_size=bn_size,
                     drop_rate=drop_rate,
                     num_classes=num_classes)
    elif model_depth == 169:
        model = densenet_3d.densenet169(bn_size=bn_size,
                     drop_rate=drop_rate,
                     num_classes=num_classes)
    elif model_depth == 201:
        model = densenet_3d.densenet201(bn_size=bn_size,
                     drop_rate=drop_rate,
                     num_classes=num_classes)
    elif model_depth == 264:
        model = densenet_3d.densenet264(bn_size=bn_size,
                     drop_rate=drop_rate,
                     num_classes=num_classes)

    return model

def load_weight():
    model = generate_model(model_type='densenet_3d',model_depth=121,
                     bn_size=4,
                     drop_rate=0,
                     num_classes=1).cuda()

    weights = '../weights_dense/model14.pth'
    print('loading pretrained weight {}'.format(weights))
    model.load_state_dict(torch.load(weights))
    print("-------- pre-train weights load successfully --------")
    return model


def get_img(file_dir):
    file = file_dir + '355_36.npy'  # npy文件-
    img = np.load(file)
    image = np.reshape(img, [1, 48, 48, 48])
    image = torch.tensor(image, dtype=torch.float32)
    image = image.to(device)
    return image


model_weights = []   # 用于存储模型的权重
conv_layers = []     # 用于存储模型的卷积层本身
counter = 0  # 统计模型里共有多少个卷积层
def search_for_convs(module):
    global counter
    for child in module.children():
        # 如果子模块是一个卷积层
        if isinstance(child, nn.Conv3d):
            counter += 1
            model_weights.append(child.weight)
            conv_layers.append(child)
        # 否则，继续递归搜索子模块的子模块
        search_for_convs(child)

# 从模型的最顶层开始搜索
model = load_weight()
search_for_convs(model)

print(f"Total convolution layers: {counter}")

# 确保image是4D的，例如：(1, 1, 48, 48, 48)
image = get_img(file_dir)
if len(image.shape) == 4:
    image = image.unsqueeze(0)  # 添加一个批次维度

# 定义一个hook函数来存储输出
outputs = []
def hook_fn(module, input, output):
    outputs.append(output)

# 附加hook到每个卷积层
hooks = []
for layer in conv_layers:
    hooks.append(layer.register_forward_hook(hook_fn))

# 前向传递图像通过模型
_ = model(image)

# 删除hooks
for hook in hooks:
    hook.remove()

# 输出每个特征图的形状
print(f"Total number of feature maps: {len(outputs)}")
for feature_map in outputs:
    print(feature_map.shape)
# 输出总特征
total_features = sum([feature_map.size(1) for feature_map in outputs])
print(f"Total number of features: {total_features}")


def feature_maps(outputs, save_dir=save_dir):
    # 创建保存目录（如果它不存在）
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, feature_map in enumerate(outputs):
        # 计算所有通道的平均值
        avg_feature_map = feature_map.mean(dim=1, keepdim=True)

        # 选择中间的切片进行可视化
        slice_idx = avg_feature_map.shape[2] // 2
        slice = avg_feature_map[0, 0, slice_idx].detach().cpu().numpy()

        # 显示并保存图片
        plt.figure(figsize=(6, 6))
        plt.imshow(slice, cmap='viridis')  # viridis\gray
        # plt.colorbar()
        # plt.title(f"Feature map {idx + 1} (average)")
        save_path = os.path.join(save_dir, f"feature_map_{idx + 1}.png")
        plt.savefig(save_path)
        plt.close()


feature_maps(outputs)