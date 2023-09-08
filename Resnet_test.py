from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import time
import torch.nn as nn
from torch.utils.data import Dataset
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.models as models
import torchsummary
import time
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
import csv


from models import resnet
# Define CSV file paths
train_csv_path = "../data/train_data.csv"
val_csv_path = "../data/val_data.csv"
test_csv_path = "../data/test_data.csv"
summaryWriter = SummaryWriter("./logs/")

class LungNoduleCSVDataSet(Dataset):
    def __init__(self, csv_file):
        # 读取CSV文件
        self.dataframe = pd.read_csv(csv_file)
        self.img_paths = self.dataframe['img_path'].tolist()
        self.labels = self.dataframe['label'].tolist()
        # self.input_D = input_D
        # self.input_H = input_H
        # self.input_W = input_W

    def __len__(self):
        return len(self.dataframe)

    # def __reshape__(self,):

    def __getitem__(self, idx):
        # 加载图像数据并转换为torch tensor
        img = np.load(self.img_paths[idx])
        img_array = np.reshape(img, [1, 48, 48, 48])
        label = self.labels[idx]
        return torch.tensor(img_array, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


# 创建数据集实例
train_dataset = LungNoduleCSVDataSet(train_csv_path)
val_dataset = LungNoduleCSVDataSet(val_csv_path)
test_dataset = LungNoduleCSVDataSet(test_csv_path)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device =',device)
print(torch.cuda.get_device_name(0))


def generate_model(model_type='resnet', model_depth=50,
                   input_W=48, input_H=48, input_D=48, resnet_shortcut='B',
                   no_cuda=False, gpu_id=[0],
                   pretrain_path = '../models/resnet_50.pth',
                   nb_class=1):
    assert model_type in [
        'resnet'
    ]

    if model_type == 'resnet':
        assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = resnet.resnet10(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 256
    elif model_depth == 18:
        model = resnet.resnet18(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 512
    elif model_depth == 34:
        model = resnet.resnet34(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 512
    elif model_depth == 50:
        model = resnet.resnet50(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 2048
    elif model_depth == 101:
        model = resnet.resnet101(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 2048
    elif model_depth == 152:
        model = resnet.resnet152(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 2048
    elif model_depth == 200:
        model = resnet.resnet200(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut,
            no_cuda=no_cuda,
            num_seg_classes=1)
        fc_input = 2048

    model.conv_seg = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(),
                                   nn.Linear(in_features=fc_input, out_features=nb_class, bias=True))

    if not no_cuda:
        if len(gpu_id) > 1:
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=gpu_id)
            net_dict = model.state_dict()
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id[0])
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=None)
            net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()

    print('loading pretrained model {}'.format(pretrain_path))
    pretrain = torch.load(pretrain_path)
    pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
    # k 是每一层的名称，v是权重数值
    net_dict.update(pretrain_dict) #字典 dict2 的键/值对更新到 dict 里。
    model.load_state_dict(net_dict) #model.load_state_dict()函数把加载的权重复制到模型的权重中去

    print("-------- pre-train model load successfully --------")

    return model


model = generate_model(model_type='resnet', model_depth=50,
                   input_W=28, input_H=28, input_D=28, resnet_shortcut='B',
                   no_cuda=False, gpu_id=[0],
                   nb_class=1)

weights='../weights/mmodel19.pth'
print('loading pretrained weight {}'.format(weights))
model.load_state_dict(torch.load(weights))
print("-------- pre-train weights load successfully --------")


def test_save(model, test_loader, device):
    model.eval()  # Set model to evaluation mode

    n = len(test_loader)
    ACC, P, R, F1, total_TN, total_FP, total_FN, total_TP = 0, 0, 0, 0, 0, 0, 0, 0

    metrics_dict = {
        "0": {"ACC": 0, "P": 0, "R": 0, "F1": 0},
        "1": {"ACC": 0, "P": 0, "R": 0, "F1": 0}
    }

    with torch.no_grad():
        for x, label in tqdm(test_loader):
            x = x.float().to(device)
            label = label.to(device)

            logits = model(x)
            logits = logits.reshape([label.cpu().numpy().shape[0]])
            prob_out = nn.Sigmoid()(logits)

            preds = (prob_out > 0.5).float().cpu().numpy()

            tn, fp, fn, tp = confusion_matrix(label.cpu().numpy().flatten(), preds.flatten()).ravel()

            total_TN += tn
            total_FP += fp
            total_FN += fn
            total_TP += tp

            all = tn + fp + fn + tp

            ACC += (tp + tn) / all
            P += tp / (tp + fp + 1e-10)
            R += tp / (tp + fn + 1e-10)

    ACC /= n
    P /= n
    R /= n
    F1 = 2 * P * R / (P + R + 1e-10)

    # Store metrics for class 0 and 1
    metrics_dict["0"]["ACC"] = total_TN / (total_TN + total_FN + 1e-10)
    metrics_dict["0"]["P"] = total_TN / (total_TN + total_FN + 1e-10)
    metrics_dict["0"]["R"] = total_TN / (total_TN + total_FP + 1e-10)
    metrics_dict["0"]["F1"] = 2 * metrics_dict["0"]["P"] * metrics_dict["0"]["R"] / (metrics_dict["0"]["P"] + metrics_dict["0"]["R"] + 1e-10)

    metrics_dict["1"]["ACC"] = ACC
    metrics_dict["1"]["P"] = P
    metrics_dict["1"]["R"] = R
    metrics_dict["1"]["F1"] = F1

    # Save evaluation metrics to a CSV file
    with open("../data/evaluation_metrics.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Class", "ACC", "P", "R", "F1"])
        for cls, metrics in metrics_dict.items():
            writer.writerow([cls, metrics["ACC"], metrics["P"], metrics["R"], metrics["F1"]])

    # Visualization of the total confusion matrix
    classes = ["normal", "nodule"]
    confusion = [[total_TN, total_FP], [total_FN, total_TP]]
    plt.figure(figsize=(12, 10))
    plt.imshow(confusion, cmap=plt.cm.Blues)
    plt.xticks(range(len(classes)), classes, fontsize=18)
    plt.yticks(range(len(classes)), classes, fontsize=18)
    plt.ylim(1.5, -0.5)
    plt.title("Confusion matrix for ResNet50", fontdict={'weight': 'normal', 'size': 18})
    # cb = plt.colorbar()
    # cb.ax.tick_params(labelsize=18)
    # plt.xlabel('Predict label', fontsize=18)
    # plt.ylabel('True label', fontsize=18)
    for i in range(len(confusion)):
        for j in range(len(confusion[i])):
            color = "w" if confusion[i][j] > 10 else "black"
            plt.text(j, i, confusion[i][j], fontsize=18, color=color, verticalalignment='center', horizontalalignment='center')
    plt.show()

    return metrics_dict, total_TN, total_FP, total_FN, total_TP

# 使用上面的函数来评估模型
model = model.to(device)
# accuracy, precision, recall, f1 = test(model, test_loader, device)
test_save(model, test_loader, device)



