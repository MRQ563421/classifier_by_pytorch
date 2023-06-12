# from torch.utils.data import Dataset
# from PIL import Image
# import os
#
# class MyDataSet(Dataset):
#
#     def __init__(self,root_path,label_path):
#         self.root_path = root_path
#         self.label_path = label_path
#         self.path = os.path.join(root_path,label_path)   # 将两个路径拼接起来形成真实路径
#         self.img_dir = os.listdir(self.path)           # 将path路径下的信息提取出来形成list，这里的信息是文件名
#
#     # 根据索引建立图片对象，并返回对象和label信息
#     def __getitem__(self, idx):
#         img_name = self.img_dir[idx]
#
#         img_path = os.path.join(self.root_path,self.label_path,img_name) # 获取单个图片的路径
#         # img_path = os.path.join(self.path, img_name) 是错误的
#         img = Image.open(img_path)  # 调用Image.open()，获取文件对象
#
#         label = self.label_path
#         return img, label
#
#     def __len__(self):
#         return len(self.img_dir)
#
# root = "D:\\pythonProject1\\mydataset\\train"
# label_ants = "ants"
# mydataset_ants = MyDataSet(root,label_ants)
#
# label_bees = "bees"
# mydataset_bees = MyDataSet(root,label_bees)
#
# mydata = mydataset_bees + mydataset_ants  # 数据集合之间可以直接相加合并
# img,label = mydata[200] # 取出数据集合中的某一具体的数据及其label
#
# print(img)
# print(label)
#
# # 我们读取图片的根目录， 在根目录下有所有图片的txt文件， 拿到txt文件后， 先读取txt文件， 之后遍历txt文件中的每一行， 首先去除掉尾部的换行符， 在以空格切分，前半部分是图片名称， 后半部分是图片标签， 当图片名称和根目录结合，就得到了我们的图片路径
# import os
#
# import numpy as np
# import torch
# import torchvision
# from PIL import Image
# from matplotlib import pyplot as plt
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
#
# transforms = transforms.Compose([
#     transforms.Resize(256),  # 将图片短边缩放至256，长宽比保持不变：
#     transforms.CenterCrop(224),  # 将图片从中心切剪成3*224*224大小的图片
#     transforms.ToTensor()  # 把图片进行归一化，并把数据转换成Tensor类型
# ])
#
#
# class MyDataset(Dataset):
#     def __init__(self, img_path, transform=None):
#         super(MyDataset, self).__init__()
#         self.root = img_path
#
#         self.txt_root = self.root + '\\' + 'data.txt'
#
#         f = open(self.txt_root, 'r')
#         data = f.readlines()
#
#         imgs = []
#         labels = []
#         for line in data:
#             line = line.rstrip()
#             word = line.split()
#             # print(word[0], word[1], word[2])
#             # word[0]是图片名字.jpg  word[1]是label  word[2]是文件夹名，如sunflower
#             imgs.append(os.path.join(self.root, word[2], word[0]))
#
#             labels.append(word[1])
#         self.img = imgs
#         self.label = labels
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.label)
#
#     def __getitem__(self, item):
#         img = self.img[item]
#         label = self.label[item]
#
#         img = Image.open(img).convert('RGB')
#
#         # 此时img是PIL.Image类型   label是str类型
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         label = np.array(label).astype(np.int64)
#         label = torch.from_numpy(label)
#
#         return img, label


import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from prettytable import PrettyTable
from torchvision import datasets
from torchvision.models import MobileNetV2
from torchvision.transforms import transforms


class ConfusionMatrix(object):
    """
    注意版本问题,使用numpy来进行数值计算的
    """

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[t, p] += 1

    # 行代表预测标签 列表示真实标签

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("acc is", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.fields_names = ["", "pre", "recall", "spec"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            pre = round(TP / (TP + FP), 3)  # round 保留三位小数
            recall = round(TP / (TP + FN), 3)
            spec = round(TN / (FP + FN), 3)
            table.add_row([self.labels[i], pre, recall, spec])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)  # 颜色变化从白色到蓝色

        # 设置 x  轴坐标 label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 将原来的 x 轴的数字替换成我们想要的信息 self.num_classes  x 轴旋转45度
        # 设置 y  轴坐标 label
        plt.yticks(range(self.num_classes), self.labels)

        # 显示 color bar  可以通过颜色的密度看出数值的分布
        plt.colorbar()
        plt.xlabel("true_label")
        plt.ylabel("Predicted_label")
        plt.title("ConfusionMatrix")

        # 在图中标注数量 概率信息
        thresh = matrix.max() / 2
        # 设定阈值来设定数值文本的颜色 开始遍历图像的时候一般是图像的左上角
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 这里矩阵的行列交换，因为遍历的方向 第y行 第x列
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        # 图形显示更加的紧凑
        plt.show()


if __name__ == '__main__':

    print(123)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # 使用验证集的预处理方式

    data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_loot = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    # get data root path
    image_path = data_loot + "/data_set/flower_data/"
    # flower data set path

    validate_dataset = datasets.ImageFolder(root=image_path + "eval",
                                            transform=data_transform)

    batch_size = 16
    validate_loader = torch.utils.data.DataLoder(validate_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=2)

    net = MobileNetV2(num_classes=5)
    # 加载预训练的权重
    model_weight_path = "./MobileNetV2.pth"
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.to(device)

    # read class_indict
    try:
        json_file = open('./class_indicts.json', 'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    labels = [label for _, label in class_indict.item()]
    # 通过json文件读出来的label
    confusion = ConfusionMatrix(num_classes=5, labels=labels)
    net.eval()
    # 启动验证模式
    # 通过上下文管理器  no_grad  来停止pytorch的变量对梯度的跟踪
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            # 获取概率最大的元素
            confusion.update(outputs.numpy(), val_labels.numpy())
            # 预测值和标签值
    confusion.plot()
    # 绘制混淆矩阵
    confusion.summary()
    # 来打印各个指标信息































