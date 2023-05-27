import numpy as np
from layers import conv_fast
from layers import pooling
from layers import activate
from layers import fc
from layers import loss
from layers import batch_normal

class LeNet5_mini():
    def __init__(self):
        self.conv1 = conv_fast.conv((6, 1, 5, 5), stride=1, padding='SAME', bias=True, requires_grad=True)
        self.pooling1 = pooling.Maxpooling(kernel_size=(2, 2), stride=2)
        self.BN1 = batch_normal.BN(6, moving_decay=0.9, is_train=True)
        self.relu1 = activate.Relu()

        self.conv2 = conv_fast.conv((16, 6, 5, 5), stride=1, padding="VALID", bias=True, requires_grad=True)
        self.pooling2 = pooling.Maxpooling(kernel_size=(2, 2), stride=2)
        self.BN2 = batch_normal.BN(16, moving_decay=0.9, is_train=True)
        self.relu2 = activate.Relu()

        self.fc4 = fc.fc(16*14*14, 500, bias=True, requires_grad=True)
        self.relu4 = activate.Relu()
        self.fc5 = fc.fc(500, 50, bias=True, requires_grad=True)

        self.softmax = loss.softmax()

    def forward(self, imgs, labels, is_train=True):
        """
        :param imgs:输入的图片：[N,C,H,W]
        :param labels:
        :return:
        """
        x = self.conv1.forward(imgs)
        x = self.pooling1.forward(x)
        x = self.BN1.forward(x, is_train)
        x = self.relu1.forward(x)

        x = self.conv2.forward(x)
        x = self.pooling2.forward(x)
        x = self.BN2.forward(x, is_train)
        x = self.relu2.forward(x)

        # x = self.conv3.forward(x)

        x = self.fc4.forward(x)
        x = self.relu4.forward(x)
        x = self.fc5.forward(x)

        loss = self.softmax.calculate_loss(x, labels)
        prediction = self.softmax.prediction_func(x)
        return loss, prediction


    def backward(self, lr):
        """
        :param lr:学习率
        :return:
        """
        eta = self.softmax.gradient()

        eta = self.fc5.backward(eta, lr)
        eta = self.relu4.backward(eta)
        eta = self.fc4.backward(eta, lr)

        # eta = self.conv3.backward(eta, lr)

        eta = self.relu2.backward(eta)  # 激活层没有参数，不需要学习
        eta = self.BN2.backward(eta, lr)
        eta = self.pooling2.backward(eta)     # 池化层没有参数，不需要学习
        eta = self.conv2.backward(eta, lr)

        eta = self.relu1.backward(eta)
        eta = self.BN1.backward(eta, lr)
        eta = self.pooling1.backward(eta)
        eta = self.conv1.backward(eta, lr)

class LeNet5_mini_Improved():
    def __init__(self):
        self.conv1 = conv_fast.conv((6, 1, 3, 3), stride=1, padding='SAME', bias=True, requires_grad=True)
        self.pooling1 = pooling.Maxpooling(kernel_size=(2, 2), stride=2)
        self.BN1 = batch_normal.BN(6, moving_decay=0.9, is_train=True)
        self.relu1 = activate.Relu()

        self.conv2 = conv_fast.conv((16, 6, 3, 3), stride=1, padding="VALID", bias=True, requires_grad=True)
        self.pooling2 = pooling.Maxpooling(kernel_size=(2, 2), stride=2)
        self.BN2 = batch_normal.BN(16, moving_decay=0.9, is_train=True)
        self.relu2 = activate.Relu()

        self.conv3 = conv_fast.conv((26, 16, 3, 3), stride=1, padding="VALID", bias=True, requires_grad=True)
        # self.pooling3 = pooling.Maxpooling(kernel_size=(2, 2), stride=2)
        self.BN3 = batch_normal.BN(26, moving_decay=0.9, is_train=True)
        self.relu3 = activate.Relu()

        self.fc4 = fc.fc(26*13*13, 468, bias=True, requires_grad=True)
        self.relu4 = activate.Relu()
        self.fc5 = fc.fc(468, 50, bias=True, requires_grad=True)

        self.softmax = loss.softmax()

    def forward(self, imgs, labels, is_train=True):
        """
        :param imgs:输入的图片：[N,C,H,W]
        :param labels:
        :return:
        """
        x = self.conv1.forward(imgs)
        x = self.pooling1.forward(x)
        x = self.BN1.forward(x, is_train)
        x = self.relu1.forward(x)

        x = self.conv2.forward(x)
        x = self.pooling2.forward(x)
        x = self.BN2.forward(x, is_train)
        x = self.relu2.forward(x)

        x = self.conv3.forward(x)
        # x = self.pooling3.forward(x)
        x = self.BN3.forward(x, is_train)
        x = self.relu3.forward(x)

        x = self.fc4.forward(x)
        x = self.relu4.forward(x)
        x = self.fc5.forward(x)

        loss = self.softmax.calculate_loss(x, labels)
        prediction = self.softmax.prediction_func(x)
        return loss, prediction


    def backward(self, lr):
        """
        :param lr:学习率
        :return:
        """
        eta = self.softmax.gradient()

        eta = self.fc5.backward(eta, lr)
        eta = self.relu4.backward(eta)
        eta = self.fc4.backward(eta, lr)

        eta = self.relu3.backward(eta)  # 激活层没有参数，不需要学习
        eta = self.BN3.backward(eta, lr)
        # eta = self.pooling3.backward(eta)     # 池化层没有参数，不需要学习
        eta = self.conv3.backward(eta, lr)

        eta = self.relu2.backward(eta)  # 激活层没有参数，不需要学习
        eta = self.BN2.backward(eta, lr)
        eta = self.pooling2.backward(eta)     # 池化层没有参数，不需要学习
        eta = self.conv2.backward(eta, lr)

        eta = self.relu1.backward(eta)
        eta = self.BN1.backward(eta, lr)
        eta = self.pooling1.backward(eta)
        eta = self.conv1.backward(eta, lr)
