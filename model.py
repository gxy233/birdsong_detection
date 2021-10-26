import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm

def accuracy(outs, labels):
    _, preds = torch.max(outs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ModelBase(nn.Module):

    # defines mechanism when training each batch in dl
    def train_step(self, batch):
        xb, labels = batch
        outs = self(xb)
        loss = F.cross_entropy(outs, labels)
        return loss

    # similar to `train_step`, but includes acc calculation & detach
    def val_step(self, batch):
        xb, labels = batch
        outs = self(xb)
        loss = F.cross_entropy(outs, labels)
        acc = accuracy(outs, labels)
        return {'loss': loss.detach(), 'acc': acc.detach()}

    # average out losses & accuracies from validation epoch
    def val_epoch_end(self, outputs):
        batch_loss = [x['loss'] for x in outputs]
        batch_acc = [x['acc'] for x in outputs]
        avg_loss = torch.stack(batch_loss).mean()
        avg_acc = torch.stack(batch_acc).mean()
        return {'avg_loss': avg_loss, 'avg_acc': avg_acc}

    # print all data once done
    def epoch_end(self, epoch, avgs, test=False):
        s = 'test' if test else 'val'
        print(f'Epoch #{epoch + 1}, {s}_loss:{avgs["avg_loss"]}, {s}_acc:{avgs["avg_acc"]}')


@torch.no_grad()
def evaluate(model, val_dl):
    # eval mode
    model.eval()
    outputs = [model.val_step(batch) for batch in val_dl]
    return model.val_epoch_end(outputs)


def fit(epochs, lr, model, train_dl, val_dl, opt_func=torch.optim.Adam):
    torch.cuda.empty_cache()
    history = []
    # define optimizer
    optimizer = opt_func(model.parameters(), lr)
    # for each epoch...
    for epoch in range(epochs):
        # training mode
        model.train()
        # (training) for each batch in train_dl...
        for batch in tqdm(train_dl):
            # pass thru model
            loss = model.train_step(batch)
            # perform gradient descent
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # validation
        res = evaluate(model, val_dl)
        # print everything useful
        model.epoch_end(epoch, res, test=False)
        # append to history
        history.append(res)
    return history




#定义BasicBlock
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsaple=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups !=1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        #下面定义BasicBlock中的各个层
        self.conv1 = nn.Conv2d(inplanes, planes,stride=stride, kernel_size=(3,1), padding=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True) #inplace为True表示进行原地操作，一般默认为False，表示新建一个变量存储操作
        self.conv2 =  nn.Conv2d(inplanes, planes,stride=stride, kernel_size=(3,1), padding=1)
        self.bn2 = norm_layer(planes)
        self.dowansample = downsaple
        self.stride = stride

    #定义前向传播函数将前面定义的各层连接起来
    def forward(self, x):
        identity = x #这是由于残差块需要保留原始输入

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.dowansample is not None: #这是为了保证原始输入与卷积后的输出层叠加时维度相同
            identity = self.dowansample(x)

        out += identity
        out = self.relu(out)

        return out

def segment(input):    #lstm前的预处理
    output = torch.split(input, 3, dim=1)      #按时间维度切割
    return output


class LSTMnet(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(LSTMnet, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.linear = nn.Linear(hidden_dim, n_class)

    def forward(self, x):  # x's shape (batch_size, 序列长度, 序列中每个数据的长度)
        out, _ = self.lstm(x)  # out's shape (batch_size, 序列长度, hidden_dim)
        out = out[:, -1, :]  # 中间的序列长度取-1，表示取序列中的最后一个数据，这个数据长度为hidden_dim，
        # 得到的out的shape为(batch_size, hidden_dim)
        out = self.linear(out)  # 经过线性层后，out的shape为(batch_size, n_class)
        return out


import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_EDVICES"] = '9'  # 指定代号为9的那块GPU




class LSTMnet(nn.Module):
        def __init__(self, in_dim, hidden_dim, n_layer):
            super(LSTMnet, self).__init__()
            self.n_layer = n_layer
            self.hidden_dim = hidden_dim
            self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        def forward(self, x):  # x‘s shape (batch_size, 序列长度, 序列中每个数据的长度)
            out,(h_n, c_n) = self.lstm(x)  # out‘s shape (batch_size, 序列长度, hidden_dim)
            out = out[:, -1, :]  # 中间的序列长度取-1，表示取序列中的最后一个数据，这个数据长度为hidden_dim，
            # 得到的out的shape为(batch_size, hidden_dim)
            # out = self.linear(out)  # 经过线性层后，out的shape为(batch_size, n_class)
            return out,(h_n,c_n)

        def trainLstm(data,label,in_dim):                #输入feature map ，data是feature map列表,label是标签列表
            # 超参数设定
            epoch = 2
            lr = 0.01
            batch_size = 50

            torch.cuda.set_device(0)
            model = LSTMnet(in_dim, 56, 2)  # 图片大小28*28，lstm的每个隐藏层56（自己设定数量大小）个节点，2层隐藏层
            if torch.cuda.is_available():
                model = model.cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()


            points_tulpe = list(zip(data, label))

            # training and testing
            # for epoch in range(2):
                # for iteration, (train_x, train_y) in enumerate(train_loader):  # train_x‘s shape (BATCH_SIZE,1,28,28)
                # for  train_x,train_y in :  # train_x‘s shape (BATCH_SIZE,1,28,28)
            train_x=data
            train_y=label
            train_x = train_x.squeeze()  # after squeeze, train_x‘s shape (BATCH_SIZE,28,28),
            # print(train_x.size())  # 第一个28是序列长度(看做句子的长度)，第二个28是序列中每个数据的长度(词纬度)。
            train_x = train_x.cuda()
            #print(train_x[0])
            train_y = train_y.cuda()
            train_x = train_x.cuda()
            # print(test_x[0])
            # test_y = test_y.cuda()
            output,(h_n, c_n) = model(train_x)
            loss = criterion(output, train_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients


            return output,(h_n, c_n)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down,self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Classifier(ModelBase):    #UL-net
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()                                      # 1 x 128 x 24
        # self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)  # 4 x 128 x 24
        # self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)  # 8 x 128 x 24
        # self.bm1 = nn.MaxPool2d(2)                              # 8 x 64 x 12
        # self.conv3 = nn.Conv2d(8, 8, kernel_size=3, padding=1)  # 8 x 64 x 12
        # self.bm2 = nn.MaxPool2d(2)                              # 8 x 32 x 6
        # self.fc1 = nn.Linear(8*32*6, 64)
        # self.fc2 = nn.Linear(64, 2)
        # self.mid_channels=None

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.resblock=BasicBlock()
        self.lstm=LSTMnet()



    def forward(self, xb):

        out = F.relu(self.resblock(xb))
        out = F.relu(self.lstm(out))
        nn.BatchNorm2d(self.mid_channels)
        nn.ReLU(inplace=True)
        out = self.bm1(out)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)



        out = F.relu(self.conv1(xb))
        out = F.relu(self.conv2(out))
        out = self.bm1(out)
        out = F.relu(self.conv3(out))
        out = self.bm2(out)
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


model = Classifier()
print(model)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
torch.backends.cudnn.benchmark = True
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)