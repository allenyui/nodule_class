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


from models import resnet
# Define CSV file paths
train_csv_path = "../data/train_data.csv"
val_csv_path = "../data/val_data.csv"
test_csv_path = "../data/test_data.csv"
summaryWriter = SummaryWriter("./logs/resnet50/")

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


def main():
    model = generate_model(model_type='resnet', model_depth=50,
                   input_W=48, input_H=48, input_D=48, resnet_shortcut='B',
                   no_cuda=False, gpu_id=[0],
                   nb_class=1)

    weights='../weights_resnet/model4.pth'
    print('loading pretrained weight {}'.format(weights))
    model.load_state_dict(torch.load(weights))
    print("-------- pre-train weights load successfully --------")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0])).cuda() #分类不均衡 #Binary Cross Entropy Loss
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    num_epochs = 5


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

        filepath = "../weights_resnet/"
        folder = os.path.exists(filepath)
        if not folder:
            # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(filepath)
        path = '../weights_resnet/model_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), path)

    summaryWriter.close()


if __name__ == '__main__':
    main()
