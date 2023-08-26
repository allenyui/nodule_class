from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
# import medmnist
# from medmnist import INFO, Evaluator
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
from models import densenet_3d
# from densenet_3d import DenseNet

# Define CSV file paths
train_csv_path = "../data/train_data.csv"
val_csv_path = "../data/val_data.csv"
test_csv_path = "../data/test_data.csv"
summaryWriter = SummaryWriter("./logs/dense/dropout0/")

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


def main():
    # model = generate_model(model_type='resnet', model_depth=50,
    #                input_W=48, input_H=48, input_D=48, resnet_shortcut='B',
    #                no_cuda=False, gpu_id=[0],
    #                nb_class=1)
    model = generate_model(model_type='densenet_3d',model_depth=121,
                     bn_size=4,
                     drop_rate=0,
                     num_classes=1).cuda()

    # weights = '../denseweights/mmodel14.pth'
    # print('loading pretrained weight {}'.format(weights))
    # model.load_state_dict(torch.load(weights))
    # print("-------- pre-train weights load successfully --------")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0])).cuda() #分类不均衡 #Binary Cross Entropy Loss
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    num_epochs =10


    total_step = len(train_loader)
    time_list = []
    for epoch in range(num_epochs):
        start = time.time()
        per_epoch_loss = 0
        num_correct = 0
        score_list = []
        label_list = []

        val_num_correct = 0
        val_score_list = []
        val_label_list = []

        model.train()
        with torch.enable_grad():
            for x, label in tqdm(train_loader):
                x = x.float()
                x = x.to(device)
                label = label.to(device)
                label = torch.squeeze(label)
                label_list.extend(label.cpu().numpy())
                # print(label_list)

                # Forward pass
                logits = model(x)
                logits = logits.reshape([label.cpu().numpy().shape[0]])
                prob_out = nn.Sigmoid()(logits)
                # print(logits.shape)

                pro_list = prob_out.detach().cpu().numpy()
                # print(pro_list)
                # print(abc)
                # print(pro_list)
                for i in range(pro_list.shape[0]):
                    if (pro_list[i] > 0.5) == label.cpu().numpy()[i]:
                        num_correct += 1

                score_list.extend(pro_list)

                loss = criterion(logits, label.float())

                per_epoch_loss += loss.item()

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # pred = logits.argmax(dim=1)
                # num_correct += torch.eq(pred, label).sum().float().item()

            score_array = np.array(score_list)
            label_array = np.array(label_list)
            fpr_keras_1, tpr_keras_1, thresholds_keras_1 = roc_curve(label_array, score_array)
            auc_keras_1 = auc(fpr_keras_1, tpr_keras_1)

            print("Train EVpoch: {}\t Loss: {:.6f}\t Acc: {:.6f} AUC: {:.6f} ".format(epoch,
                                                                                  per_epoch_loss / len(train_loader),
                                                                                  num_correct / len(
                                                                                      train_loader.dataset),
                                                                                  auc_keras_1))
            summaryWriter.add_scalars('loss', {"loss": (per_epoch_loss / len(train_loader))}, epoch)
            summaryWriter.add_scalars('acc', {"acc": num_correct / len(train_loader.dataset)}, epoch)
            summaryWriter.add_scalars('auc', {"auc": auc_keras_1}, epoch)

        model.eval()
        with torch.no_grad():
            for x, label in tqdm(val_loader):
                x = x.float()
                x = x.to(device)
                label = label.to(device)
                # label_n = label.cpu().numpy()

                val_label_list.extend(label.cpu().numpy())

                # Forward pass
                logits = model(x)
                logits = logits.reshape([label.cpu().numpy().shape[0]])
                prob_out = nn.Sigmoid()(logits)
                # print(logits.shape)

                pro_list = prob_out.detach().cpu().numpy()

                # print(pro_list)
                for i in range(pro_list.shape[0]):
                    if (pro_list[i] > 0.5) == label.cpu().numpy()[i]:
                        val_num_correct += 1

                val_score_list.extend(pro_list)

            score_array = np.array(val_score_list)
            label_array = np.array(val_label_list)
            fpr_keras_1, tpr_keras_1, thresholds_keras_1 = roc_curve(label_array, score_array)
            auc_keras_1 = auc(fpr_keras_1, tpr_keras_1)

            print("val Epoch: {}\t Acc: {:.6f} AUC: {:.6f} ".format(epoch, val_num_correct / len(val_loader.dataset),
                                                                auc_keras_1))
            summaryWriter.add_scalars('acc', {"val_acc": val_num_correct / len(val_loader.dataset)}, epoch)
            summaryWriter.add_scalars('auc', {"val_auc": auc_keras_1}, epoch)
            summaryWriter.add_scalars('time', {"time": (time.time() - start)}, epoch)

        scheduler.step()

        filepath = "../weights_dense/drop0/"
        folder = os.path.exists(filepath)
        if not folder:
            # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(filepath)
        path = '../weights_dense/drop0/model' + str(epoch) + '.pth'
        torch.save(model.state_dict(), path)

    summaryWriter.close()


if __name__ == '__main__':
    main()
