
import os
import numpy as np
import torchvision

from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MyDataSet(Dataset):

    def __init__(self,root_path,label, transform = None):
        super(MyDataSet, self).__init__()

        self.root_path = root_path
        self.label = label
        self.path = os.path.join(root_path,label)   # 将两个路径拼接起来形成真实路径
        self.img_dir = os.listdir(self.path)           # 将path路径下的信息提取出来形成list，这里的信息是文件名

        self.transform = transform

    # 根据索引建立图片对象，并返回对象和label信息
    def __getitem__(self, idx):
        img_name = self.img_dir[idx]  # 根据索引获得文件名

        img_path = os.path.join(self.root_path,self.label,img_name) # 获取单个图片的路径
        # img_path = os.path.join(self.path, img_name) 是错误的
        img = Image.open(img_path).convert('RGB') # 调用Image.open()，获取文件对象

        if self.transform is not None:
                img = self.transform(img)

        label = self.label

        # # 将 tuple 类型的数据转换成 tensor  1是蚂蚁，2是蜜蜂
        # if label == "ants":
        #     label = 1
        # else:
        #     label= 0

        return img, label

    def __len__(self):
        return len(self.img_dir)


if __name__ == '__main__':
        trans = transforms.Compose([
            transforms.Resize(256),    # 将图片短边缩放至256，长宽比保持不变：
            transforms.CenterCrop(224),   # 将图片从中心切剪成3*224*224大小的图片
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])   # 用来调整信道3个
            transforms.ToTensor(),  # 把图片进行归一化，并把数据转换成Tensor类型

        ])

        root = "D:\\pythonProject1\\mydataset\\train"
        label_ants = "ants"
        mydataset_ants = MyDataSet(root, label_ants,transform=trans)

        label_bees = "bees"
        mydataset_bees = MyDataSet(root, label_bees,transform=trans)

        # 数据集
        mydata = mydataset_bees + mydataset_ants  # 数据集合之间可以直接相加合并
        # img, label = mydata[200]  # 取出数据集合中的某一具体的数据及其label
        # print(img)
        # print(label)

        loader = DataLoader(dataset=mydata, batch_size=16,shuffle=True)
        for data in loader:
                imgs, label = data
                print(label)
                # print(imgs)

                # # 打印数据集中的图片
                # img = torchvision.utils.make_grid(imgs).numpy()
                # plt.imshow(np.transpose(img, (1, 2, 0)))
                # plt.show()
                # break



