import datetime

import torch
import torchvision
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from prettytable import PrettyTable
from torchvision import datasets
from torchvision.models import MobileNetV2
from torchvision.transforms import transforms
from MY_DATASETS import MyDataSet



class ZFNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(ZFNet, self).__init__()
        self.features = nn.Sequential(  # 打包
            nn.Conv2d(3, 48, kernel_size=7, stride=2, padding=1),   # input[3, 224, 224]  output[48, 110, 110] 自动舍去小数点后
            nn.ReLU(inplace=True),  # inplace 可以载入更大模型
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),       # output[48, 55, 55] kernel_num为原论文一半
            nn.Conv2d(48, 128, kernel_size=5, stride=2),            # output[128, 26, 26]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),       # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            # 全连接
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # 展平   或者view()
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 何教授方法
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # 正态分布赋值
                nn.init.constant_(m.bias, 0)

class Confusion_Matrix():
    def __init__(self,num_classes: int ):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[t, p] += 1

    def calculate_accuracy(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        return acc

    def precision_recall_specificity(self):
        # precision, recall, specificity
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            pre = round(TP / (TP + FP), 3)  # round 保留三位小数
            recall = round(TP / (TP + FN), 3)
            spec = round(TN / (FP + FN), 3)
        return pre,recall,spec


class Fully_CNN(nn.Module):
    def __init__(self, class_num) -> None:
        super().__init__()

        self.model = torchvision.models.mobilenet_v2(pretrained=True)
        self.features = self.model.features
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=self.model.last_channel, out_channels=class_num, kernel_size=1, bias=False)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = self.classifier(x)
        return x.reshape((x.size(0), -1))


if __name__ == '__main__':

# a.初始化参数batchsize epoch 和类别数
    BATCH_SIZE = 32
    EPOCH = 200
    CLASS_NUM = 2
    best_acc = 0  # 记录最好的模型以便输出
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 设备

# b.加载自己的数据
    # 1.调整数据格式
    trans = transforms.Compose([
        transforms.Resize(256),  # 将图片短边缩放至256，长宽比保持不变：
        transforms.CenterCrop(224),  # 将图片从中心切剪成3*224*224大小的图片
        transforms.ToTensor(),  # 把图片进行归一化，并把数据转换成Tensor类型
    ])

    # 训练数据集
    # train_root = "D:\\pythonProject1\\mydataset\\train"
    # train_label_ants = "ants"
    train_root = "D:/pythonProject1/Mydataset1/train"
    train_label_0 = "0"
    train_mydataset_ants = MyDataSet(train_root, train_label_0, transform=trans)

    # train_label_bees = "bees"
    train_label_1 = "1"
    train_mydataset_bees = MyDataSet(train_root, train_label_1, transform=trans)

    train_mydata = train_mydataset_bees + train_mydataset_ants  # 数据集合之间可以直接相加合并
    # 数据生成器
    train_loader = DataLoader(dataset=train_mydata, batch_size=BATCH_SIZE, shuffle=True)

    # 测试数据集
    # eval_root = "D:\\pythonProject1\\mydataset\\eval"
    # eval_label_ants = "ants"
    eval_root = "D:/pythonProject1/Mydataset1/test"
    eval_label_0 = "0"
    eval_mydataset_ants = MyDataSet(eval_root,eval_label_0, transform=trans)

    # eval_label_bees = "bees"
    eval_label_1 = "1"
    eval_mydataset_bees = MyDataSet(eval_root,eval_label_1, transform=trans)

    eval_mydata = eval_mydataset_bees + eval_mydataset_ants  # 数据集合之间可以直接相加合并
    eval_loader = DataLoader(dataset=eval_mydata, batch_size=BATCH_SIZE, shuffle=False)


# b.加载自己的数据
    # 1.调整数据格式
    trans1_1 = transforms.Compose([
        transforms.Resize(256),  # 将图片短边缩放至256，长宽比保持不变：
        transforms.CenterCrop(224),  # 将图片从中心切剪成3*224*224大小的图片
        # transforms.AutoAugment(),
        transforms.ToTensor(),  # 把图片进行归一化，并把数据转换成Tensor类型
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    trans1_2 = transforms.Compose(
        [
        transforms.CenterCrop(224),  # 将图片从中心切剪成3*224*224大小的图片
        transforms.ToTensor(),  # 把图片进行归一化，并把数据转换成Tensor类型
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    # 数据生成器的定义
    train_generator = torchvision.datasets.ImageFolder('D:/pythonProject1/Mydataset1/train', transform=trans1_1)
    train_generator = torch.utils.data.DataLoader(train_generator, BATCH_SIZE, shuffle=True, num_workers=4)

    test_generator = torchvision.datasets.ImageFolder('D:\\pythonProject1\\Mydataset1\\val', transform=trans1_2)
    test_generator = torch.utils.data.DataLoader(test_generator, BATCH_SIZE, shuffle=False, num_workers=4)


# c.搭建神经网络
    module = ZFNet(CLASS_NUM)
    module1_2 = Fully_CNN(CLASS_NUM)
# d.损失函数-交叉熵损失
    loss_fn = nn.CrossEntropyLoss()

# e.优化器
    optimizer = torch.optim.Adam(module1_2.parameters(),lr =1e-4,weight_decay=5e-4)
    lr_step = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=5, eta_min=1e-6)

# 训练过程
    print('{} begin train on {}!'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), DEVICE))    # 打印开始训练信息时间和设备
    with open('train.log', 'w+') as f:
        f.write('train_loss,eval_loss,train_acc,eval_acc')     # 在train文件日志下，写下训练过程中的数据

    for epoch in range(EPOCH):
        begin = time.time()
        print("------epoch: {}轮------".format(epoch))

    # 训练步骤开始
        print("训练开始")
        module1_2.to(DEVICE)
        module1_2.train()  # 这一步只对部分训练集合有效

        # 混淆矩阵
        train_cm = Confusion_Matrix(CLASS_NUM)
        train_loss = []

        for data in train_generator:
            imgs, target = data

            pred = module1_2(imgs)
            # 计算损失函数-交叉熵
            loss = loss_fn(pred, target)

            # 进行优化
            optimizer.zero_grad()  # 优化前要将梯度置零
            loss.backward()  # 计算梯度，反向传播
            optimizer.step()  # 根据梯度进行优化参数


            # 获取概率最大的元素，以获得混淆矩阵
            outputs = torch.softmax(pred, dim=1)
            outputs = torch.argmax(outputs, dim=1)

            train_loss.append(float(loss.data))
            # 计算混淆矩阵
            train_cm.update(outputs,target)

        # 计算train_acc
        train_acc = train_cm.calculate_accuracy()
        # 训练平均损失
        train_loss = np.mean(train_loss)

    # 测试过程
        print("测试开始")
        module1_2.eval()

        # 混淆矩阵
        eval_cm = Confusion_Matrix(CLASS_NUM)
        eval_loss = []
        with torch.no_grad():
            for data in test_generator:
                imgs, target = data

                pred_eval = module1_2(imgs)
                loss_eval = loss_fn(pred_eval,target)


                # 获取概率最大的元素
                outputs_eval = torch.softmax(pred_eval, dim=1)
                outputs_eval = torch.argmax(outputs_eval, dim=1)

                eval_loss.append(float(loss_eval.data))
                # 计算混淆矩阵
                eval_cm.update(outputs,target)

        # 计算eval_acc
        eval_acc = eval_cm.calculate_accuracy()
        # 训练平均损失
        eval_loss = np.mean(eval_loss)

        # 保存结果最好的
        if eval_acc > best_acc:
            best_acc = eval_acc
            module1_2.to('cpu')
            torch.save(module1_2, 'mymodel1-2.pt')

        # 在train.log中打印信息
        with open('train.log', 'a+') as f:
            f.write('\n{:.5f},{:.5f},{:.4f},{:.4f}'.format(train_loss, eval_loss, train_acc, eval_acc))

        print('{} epoch:{}, time:{:.2f}s, train_loss:{:.5f}, eval_loss:{:.5f}, train_acc:{:.4f}, eval_acc:{:.4f}'.format(
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            epoch + 1, time.time() - begin, train_loss, eval_loss, train_acc, eval_acc
        ))