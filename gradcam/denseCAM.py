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
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from models import resnet
from models import densenet_3d

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device =', device)
print(torch.cuda.get_device_name(0))


def generate_model(model_type='densenet_3d', model_depth=121,bn_size=4,
                     drop_rate=0,
                     num_classes=1):
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

    # weights = '../denseweights/mmodel6.pth'
    weights = '../weights_dense/model14.pth'
    print('loading pretrained weight {}'.format(weights))
    model.load_state_dict(torch.load(weights))
    print("-------- pre-train weights load successfully --------")
    return model

model = load_weight()
# summary(model, input_size=(1, 48, 48, 48), batch_size=32, device='cuda')
# print(model)
file_dir = r"E:\Workplace\dataset\classification_aug1/"  # npy文件路径
# file_dir = r"E:\Workplace\dataset\classification0/"  # npy文件路径
file = file_dir + '233_34.npy'  # npy文件-
img = np.load(file)
image = np.reshape(img, [1, 48, 48, 48])
image = torch.tensor(image, dtype=torch.float32)
image = image.to(device)
# 确保image是4D的，例如：(1, 1, 48, 48, 48)
image = np.load(file)
image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)
if len(image.shape) == 4:
    image = image.unsqueeze(0)  # 添加一个批次维度


def extract_conv_layers_and_features(model, image):
    """
    Extract convolutional layers from the model and get feature maps for a given image.

    Parameters:
    - model: PyTorch model from which convolutional layers and features are to be extracted.
    - image: Image tensor to pass through the model to get feature maps.

    Returns:
    - conv_layers: List of convolutional layers in the model.
    - outputs: List of feature maps from the convolutional layers.
    """

    model_weights = []  # 用于存储模型的权重
    conv_layers = []  # 用于存储模型的卷积层本身
    counter = 0  # 统计模型里共有多少个卷积层

    def search_for_convs(module):
        nonlocal counter
        for child in module.children():
            # 如果子模块是一个卷积层
            if isinstance(child, nn.Conv3d):
                counter += 1
                model_weights.append(child.weight)
                conv_layers.append(child)
            # 否则，继续递归搜索子模块的子模块
            search_for_convs(child)

    # 从模型的最顶层开始搜索
    search_for_convs(model)

    # 定义一个hook函数来存储输出
    outputs = []

    def hook_fn(module, input, output):
        outputs.append(output)

    # 附加hook到每个卷积层
    hooks = []
    for layer in conv_layers:
        hooks.append(layer.register_forward_hook(hook_fn))

    # 前向传递图像通过模型
    model.eval()
    _ = model(image)

    # 删除hooks
    for hook in hooks:
        hook.remove()

    return conv_layers, outputs


# 使用这个函数：
conv_layers, outputs = extract_conv_layers_and_features(model, image)

# 输出每个特征图的形状
print(f"Total number of feature maps: {len(outputs)}")
for feature_map in outputs:
    print(feature_map.shape)
# 输出总特征
total_features = sum([feature_map.size(1) for feature_map in outputs])
print(f"Total number of features: {total_features}")


def compute_gradcam(model, image, conv_layers):
    # 1. 正向传播到所选层获取特征图
    features = None
    gradients = None

    def hook_fn(module, input, output):
        nonlocal features
        features = output

    def hook_gradient(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    # 注册钩子
    target_layer = conv_layers[57]  # 选择第88卷积层
    target_layer.register_forward_hook(hook_fn)
    target_layer.register_full_backward_hook(hook_gradient)

    # 2. 获取目标类得分
    output = model(image)
    # 对于二分类，获取正类得分
    score = output[0]
    print("Model's prediction:", output)
    probability = nn.Sigmoid()(score)
    print("Probability of positive class:", probability.item())
    preds = (probability > 0.5).float().cpu().numpy()

    print("Predicted labels:", preds)


    # 3. 计算梯度
    score.backward()

    # 4. 全局平均池化
    pooled_gradients = torch.mean(gradients, dim=[2, 3, 4])

    # 5. 生成激活映射
    for i in range(pooled_gradients.shape[1]):
        features[:, i, :, :, :] *= pooled_gradients[0, i]
    activation_map = torch.mean(features, dim=1).squeeze()

    # 6. ReLU激活
    activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())

    # 7. 上采样激活映射
    upsampled_activation_map = resize(activation_map.detach().cpu().numpy(),
                                      (image.shape[2], image.shape[3], image.shape[4]),
                                      order=1, mode='constant', preserve_range=True)

    # 8. 获取原始输入图像
    original_image = image.squeeze().cpu().numpy()

    return original_image, upsampled_activation_map

original_image, upsampled_activation_map = compute_gradcam(model, image, conv_layers)



def save_slices(original_volume, activation_volume, save_dir="slices_355_36_0", axis=2):
    """Save overlayed slices from a 3D volume to disk."""

    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Total number of slices along the given axis
    num_slices = original_volume.shape[axis]

    for idx in range(num_slices):
        if axis == 0:
            original_slice = original_volume[idx, :, :]
            activation_slice = activation_volume[idx, :, :]
        elif axis == 1:
            original_slice = original_volume[:, idx, :]
            activation_slice = activation_volume[:, idx, :]
        else:
            original_slice = original_volume[:, :, idx]
            activation_slice = activation_volume[:, :, idx]

        # Create the overlayed image
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(original_slice, cmap='gray')  # Display the original slice in gray
        ax.imshow(activation_slice, cmap='jet', alpha=0.5)  # Overlay the activation map
        ax.axis('off')

        # Save the image
        save_path = os.path.join(save_dir, f"overlayed_slice_{idx}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

# Save slices along z-axis to disk
save_slices(original_image, upsampled_activation_map)




