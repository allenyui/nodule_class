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
from models import densenet_3d

# Define CSV file paths
train_csv_path = "../data/train_data.csv"
val_csv_path = "../data/val_data.csv"
test_csv_path = "../data/test_data.csv"
summaryWriter = SummaryWriter("./logs_test1/")

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


model = generate_model(model_type='densenet_3d', model_depth=121,
                       bn_size=4,
                       drop_rate=0,
                       num_classes=1).cuda()

weights='../weights_dense/model14.pth'
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

            # print("Predicted labels:", preds)
            # print("True labels:", label.cpu().numpy())

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
    with open("../data/evaluation_metrics_dense1.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Class", "ACC", "P", "R", "F1"])
        for cls, metrics in metrics_dict.items():
            writer.writerow([cls, metrics["ACC"], metrics["P"], metrics["R"], metrics["F1"]])

    # Visualization of the total confusion matrix
    classes = ["normal", "nodule"]
    confusion = [[total_TN, total_FP], [total_FN, total_TP]]
    plt.figure(figsize=(12, 10))
    plt.imshow(confusion, cmap=plt.cm.Blues)
    plt.xticks(range(len(classes)), classes, rotation=40, fontsize=18)
    plt.yticks(range(len(classes)), classes, fontsize=18)
    plt.ylim(1.5, -0.5)
    plt.title("Confusion matrix for DenseNet121", fontdict={'weight': 'normal', 'size': 18})
    # cb = plt.colorbar()
    # cb.ax.tick_params(labelsize=18)
    # plt.xlabel('Predict label', fontsize=18)
    # plt.ylabel('True label', fontsize=18)
    for i in range(len(confusion)):
        for j in range(len(confusion[i])):
            color = "w" if confusion[i][j] > 400 else "black"
            plt.text(j, i, confusion[i][j], fontsize=18, color=color, verticalalignment='center', horizontalalignment='center')
    plt.show()

    return metrics_dict, total_TN, total_FP, total_FN, total_TP

# 使用上面的函数来评估模型
model = model.to(device)
# accuracy, precision, recall, f1 = test(model, test_loader, device)
test_save(model, test_loader, device)



