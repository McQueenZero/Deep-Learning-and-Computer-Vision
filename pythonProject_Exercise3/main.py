# --------------------------------------------------------------------------------------------------------------------------------------------------
# 作者：       赵敏琨
# 日期：       2021年6月
# 说明：       实验三：实践篇
# 任务1：    利用至少两种不同的神经网络进行训练，完成手写数字识别；
# 任务2：    训练过程准确度、损失函数可视化；可视化所用神经网络的结构；单张图输入，特征图可视化。
# --------------------------------------------------------------------------------------------------------------------------------------------------

# Reference: All of pythonProject_Exercise2 reference
# Reference: https://blog.csdn.net/akadiao/article/details/79761790 Python学习（二十一）——使用matplotlib交互绘图
# Reference: https://blog.csdn.net/wuzlun/article/details/80053277 Python绘图总结(Matplotlib篇)之坐标轴及刻度
# Reference: https://blog.csdn.net/sinat_37938004/article/details/106230959 Matplotlib:subplots多子图时添加主标题和子标题

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import mynet
import matplotlib.pyplot as plt
import cv2
import time
import os
import copy
import mousedraw

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#   ----------任务选择----------
print("任务1：手写数字识别模型训练，请输入'HDR'")  # Handwritten Digit Recognition
print("任务2：手写数字识别特征图可视化，请输入'FV'")  # Feature Visualization
while 1:
    TASK = input('请选择任务：')
    if TASK == 'HDR' or TASK == 'hdr':
        break
    elif TASK == 'FV' or TASK == 'fv':
        break
    else:
        print('非法，请重新输入')

#   ----------手写数字识别----------
if TASK == 'HDR' or TASK == 'hdr':

    plt.ion()  # interactive mode

    # #########Choose Net######### #
    print("选择网络MyNet，请输入：'mine'")
    print("选择网络ResNet，请输入：'res'")
    print("选择网络SqueezeNet，请输入：'squeeze'")
    while 1:
        NetName = input('请选择网络：')
        if NetName == 'mine':
            break
        elif NetName == 'res':
            break
        elif NetName == 'squeeze':
            break
        else:
            print('非法，请重新输入')

    # Dataset import
    # Data augmentation and normalization for training
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    data_dir = './data'
    image_datasets = {x: datasets.MNIST(data_dir, train=x, download=False,
                                        transform=data_transforms)
                      for x in [True, False]}
    dataloaders = {x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=500, shuffle=True)
        for x in [True, False]}
    # Here above is a problem, or bug?
    # When I set num_worker=1, the main code tried to run repeatedly.
    # I watched the debug window found that DataLoader module
    # did not throw an error but exited unexpectedly.
    # Theoretically, 'num_worker=1' should be fine, but it isn't.
    # A solution way is to delete the expression 'num_worker=1'.
    dataset_sizes = {x: len(image_datasets[x]) for x in [True, False]}
    class_names = image_datasets[True].classes

    ######################################################################
    # Training the model
    # ------------------
    #
    # Write a general function to train a model. Here, I will illustrate:
    #
    # -  Scheduling the learning rate
    # -  Saving the best model
    #
    # In the following, parameter ``scheduler`` is an LR scheduler object from
    # ``torch.optim.lr_scheduler``.

    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        fig_train, axs_train = plt.subplots(1, 2)
        fig_val, axs_val = plt.subplots(1, 2)
        fig_train.suptitle('Train Process')
        fig_val.suptitle('Val Process')
        xdata = range(num_epochs)
        loss_train, acc_train = [], []
        loss_val, acc_val = [], []

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in [True, False]:
                if phase == True:
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == True):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == True:
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == True:
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.cpu() / dataset_sizes[phase]

                if phase == True:
                    print('Train Loss: {:.4f} Acc: {:.4f}'.format(
                        epoch_loss, epoch_acc))
                    loss_train.append(epoch_loss)
                    acc_train.append(epoch_acc)
                else:
                    print('Val Loss: {:.4f} Acc: {:.4f}'.format(
                        epoch_loss, epoch_acc))
                    loss_val.append(epoch_loss)
                    acc_val.append(epoch_acc)

                # deep copy the model
                if phase == False and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        axs_train[0].grid()
        axs_train[0].plot(xdata, loss_train, '.-')
        axs_train[0].set_title('Loss Curve')
        axs_train[1].grid()
        axs_train[1].plot(xdata, acc_train, '.-')
        axs_train[1].set_title('Accuracy Curve')
        axs_train[0].set_ylabel('Value')
        axs_train[0].set_xlabel("Epoch")
        axs_train[1].set_xlabel("Epoch")
        axs_val[0].grid()
        axs_val[0].plot(xdata, loss_val, '.-')
        axs_val[0].set_title('Loss Curve')
        axs_val[1].grid()
        axs_val[1].plot(xdata, acc_val, '.-')
        axs_val[1].set_title('Accuracy Curve')
        axs_val[0].set_ylabel("Value")
        axs_val[0].set_xlabel("Epoch")
        axs_val[1].set_xlabel("Epoch")

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model


    ######################################################################
    # Visualize a few images
    # ^^^^^^^^^^^^^^^^^^^^^^
    #

    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array((0.1307,))
        std = np.array((0.3081,))
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.matshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated


    # Get a batch of training data
    batch_idx, (inputs, classes) = next(iter(enumerate(dataloaders[True])))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out)
    # imshow(out, title=str([class_names[x][0] for x in classes]))

    ######################################################################
    # Visualizing the model predictions
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #
    # Generic function to display predictions for a few images
    #

    def visualize_model(model, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders[False]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {}'.format(
                        class_names[preds[j]][0]))
                    plt.imshow(inputs.cpu().data[j][0, :, :],
                               cmap='gray')   # Gray scale image

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return model.train(mode=was_training)

    ######################################################################
    # Tuning the convnet
    # ----------------------
    #
    # Load a pretrained model and reset first & final fully connected layer.
    #

    if NetName == 'mine':
        model_t = mynet.Net()
    elif NetName == 'res':
        model_t = models.resnet18(pretrained=False)
        model_t.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2))

        num_ftrs = model_t.fc.in_features
        model_t.fc = nn.Linear(num_ftrs, 10)
    elif NetName == 'squeeze':
        model_t = models.squeezenet1_1(pretrained=False)
        model_t.features._modules['0'] = nn.Conv2d(
            1, 64, kernel_size=(3, 3), stride=(2, 2))

        num_ftrs = model_t.classifier._modules['1'].in_channels
        tuple_kernel = model_t.classifier._modules['1'].kernel_size
        tuple_stride = model_t.classifier._modules['1'].stride
        # The squeeze net has no full connection layers, needed to be changed
        model_t.classifier._modules['1'] = nn.Conv2d(
            num_ftrs, 10, tuple_kernel, tuple_stride)

    model_t = model_t.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are not being optimized
    optimizer_ft = optim.Adam(model_t.parameters(), lr=0.001, betas=(0.9, 0.999))

    if NetName == 'res':
        SS = 4
    else:
        SS = 7

    # Decay LR by a factor of 0.1 every SS epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=SS, gamma=0.1)

    ######################################################################
    # Train and evaluate
    # ^^^^^^^^^^^^^^^^^^
    #
    # It takes about two minute on GPU.
    #

    if NetName == 'res':
        NE = 13
    else:
        NE = 25

    model_t = train_model(
        model_t, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=NE)

    ######################################################################
    #

    visualize_model(model_t)

    ######################################################################
    #
    while 1:
        SAVEFLAG = input('请输入训练网络是否保存(ON/OFF)：')
        if SAVEFLAG == 'ON' or SAVEFLAG == 'on':
            break
        elif SAVEFLAG == 'OFF' or SAVEFLAG == 'off':
            break
        else:
            print('非法，请重新输入')

    if SAVEFLAG == 'ON' or SAVEFLAG == 'on':
        ff = torch.randn(500, 1, 28, 28)
        ff = ff.type(torch.cuda.FloatTensor)
        if NetName == 'mine':
            torch.save(model_t, './models/MyNet.pth')
            torch.onnx.export(model_t, ff, './nets/MyNet.onnx')
        elif NetName == 'res':
            torch.save(model_t, './models/ResNet.pth')
            torch.onnx.export(model_t, ff, './nets/ResNet.onnx')
        elif NetName == 'squeeze':
            torch.save(model_t, './models/SqueezeNet.pth')
            torch.onnx.export(model_t, ff, './nets/SqueezeNet.onnx')
        print('网络已保存')
    else:
        print('保存已关闭')

    plt.ioff()
    plt.show()

#   ----------特征图可视化----------
if TASK == 'FV' or TASK == 'fv':

    plt.ion()  # interactive mode

    # #########Choose Net & Validation Loading Way######### #
    print("选择网络MyNet，请输入：'mine'")
    print("选择网络ResNet，请输入：'res'")
    print("选择网络SqueezeNet，请输入：'squeeze'")
    while 1:
        NetName = input('请选择网络：')
        if NetName == 'mine':
            my_model = torch.load('./models/MyNet.pth')
            break
        elif NetName == 'res':
            my_model = torch.load('./models/ResNet.pth')
            break
        elif NetName == 'squeeze':
            my_model = torch.load('./models/SqueezeNet.pth')
            break
        else:
            print('非法，请重新输入')

    print("从本地加载，请输入：'LL'")   # Load from Local path
    print("交互式用鼠标画出数字，请输入：'MD'")    # Mouse Drawing
    while 1:
        TESTPATH = input('请输入测试方式：')
        if TESTPATH == 'LL' or TESTPATH == 'll':
            break
        elif TESTPATH == 'MD' or TESTPATH == 'md':
            break
        else:
            print('非法，请重新输入')

    if TESTPATH == 'LL' or TESTPATH == 'll':
        FILENUM = input('文件编号：')
        # Randomly choose an image
        # FILENUM = str(np.random.randint(0, 10, 1)[0])
        img_dir = './data/FV_test/' + FILENUM + '.jpg'
        img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
        img = img[10:110, 0:128]  # Cut and Resize(Only for my LL samples)
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        img = mousedraw.inverse_color(img)  # Reverse color
        plt.figure('Input Image')
        plt.axis('off')
        plt.imshow(img, cmap='gray')
        plt.pause(0.001)

    elif TESTPATH == 'MD' or TESTPATH == 'md':
        img = mousedraw.draw()
        plt.figure('Input Image')
        plt.axis('off')
        plt.imshow(img, cmap='gray')
        plt.pause(0.001)

    transforms_act = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    img_trans = transforms_act(img).unsqueeze(0)  # Augmentation
    img_trans = img_trans.to(device)
    outputs = my_model(img_trans)

    _, preds = torch.sort(outputs, 1, descending=True)

    preds_list = preds[0]
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    preds_top1 = class_names[preds_list[0]]
    preds_top3 = class_names[preds_list[0]], class_names[preds_list[1]], class_names[preds_list[2]]
    print("预测结果Top1: ", preds_top1)
    print('预测结果Top3:', preds_top3)

    if NetName == 'mine' or NetName == 'res':
        module_name_list = []
        print('↓-----------------------------------------------------------------------------------------------------↓')
        print('Model Architecture:')
        for module_name in my_model._modules:
            print(module_name)
            module_name_list.append(module_name)
        print('↑-----------------------------------------------------------------------------------------------------↑')

        # Feature Extractor Class
        class FeatureExtractor(nn.Module):
            def __init__(self, submodule, extracted_layers):
                super(FeatureExtractor, self).__init__()
                self.submodule = submodule
                self.extracted_layers = extracted_layers

            def forward(self, x):
                outputs = []
                for name, module in self.submodule._modules.items():
                    if NetName == 'res' and name == "fc":
                        x = x.view(x.size(0), -1)
                    if NetName == 'mine' and name == 'fc1':
                        x = x.view(-1, 4 * 4 * 50)
                    x = module(x)
                    # print(name)
                    if name in self.extracted_layers:
                        outputs.append(x)
                return outputs


        exact_list = []
        while 1:
            exact = input('请输入抽取特征层名称：')
            print(exact)
            if exact in module_name_list:
                exact_list.append(exact)
                while 1:
                    ctn = input('继续输入？(Y/N)：')
                    if ctn == 'Y' or ctn == 'y':
                        break
                    elif ctn == 'N' or ctn == 'n':
                        break
                    else:
                        print('非法，请重新输入')
                if ctn == 'N' or ctn == 'n':
                    break
            else:
                print('非法，请重新输入')

    elif NetName == 'squeeze':
        module_name_list = []
        print('↓-----------------------------------------------------------------------------------------------------↓')
        print('Model Architecture:')
        for module_name in my_model.features._modules:
            print(module_name, ': ', my_model.features[int(module_name)])
            module_name_list.append(module_name)
        print('↑-----------------------------------------------------------------------------------------------------↑')

        # Feature Extractor Class
        class FeatureExtractor(nn.Module):
            def __init__(self, submodule, extracted_layers):
                super(FeatureExtractor, self).__init__()
                self.submodule = submodule
                self.extracted_layers = extracted_layers

            def forward(self, x):
                outputs = []
                for name, module in self.submodule.features._modules.items():
                    if name == "fc":
                        x = x.view(x.size(0), -1)
                    x = module(x)
                    # print(name)
                    if name in self.extracted_layers:
                        outputs.append(x)
                return outputs


        exact_list = []
        while 1:
            exact = input('请输入抽取特征层编号：')
            print(exact)
            if exact in module_name_list:
                exact_list.append(exact)
                while 1:
                    ctn = input('继续输入？(Y/N)：')
                    if ctn == 'Y' or ctn == 'y':
                        break
                    elif ctn == 'N' or ctn == 'n':
                        break
                    else:
                        print('非法，请重新输入')
                if ctn == 'N' or ctn == 'n':
                    break
            else:
                print('非法，请重新输入')

    my_exactor = FeatureExtractor(my_model, exact_list)
    x_tensor_list = my_exactor(img_trans)

    for k in range(len(exact_list)):
        x_cpu = x_tensor_list[k].data.cpu()
        print('LAYER:' + exact_list[k] + "'s", 'output shape is', x_cpu.shape)

        if NetName == 'mine':
            if x_cpu.shape[1] < 32:
                subplot_rows = 4
            else:
                subplot_rows = 5
        else:
            if x_cpu.shape[1] < 256:
                subplot_rows = 8
            elif x_cpu.shape[1] <= 512:
                subplot_rows = 16
            else:
                subplot_rows = 32
        subplot_cols = int(x_cpu.shape[1] / subplot_rows)

        # Feature visualization method 1: Traversal
        x_fusion = np.zeros((x_cpu.shape[2], x_cpu.shape[3]))
        plt.figure('Feature Maps of LAYER:' + exact_list[k])
        for i in range(x_cpu.shape[1]):
            ax = plt.subplot(subplot_rows, subplot_cols, i + 1)
            # ax.set_title('Sample #{}'.format(i))
            ax.axis('off')
            plt.imshow(x_cpu.data.numpy()[0, i, :, :], cmap='viridis')
            x_fusion += x_cpu.data.numpy()[0, i, :, :]

        plt.figure('Random 4 Feature Maps of LAYER:' + exact_list[k])
        x_random = np.random.randint(0, x_cpu.shape[1], 4)
        for i in range(4):
            ax = plt.subplot(2, 2, i + 1)
            ax.axis('off')
            ax.set_title('Sample #{}'.format(x_random[i]))
            plt.imshow(x_cpu.data.numpy()[0, x_random[i], :, :], cmap='viridis')

        # Feature visualization method 2: Based on traversal, feature maps fusion
        plt.figure('Fusion Feature Map of LAYER:' + exact_list[k])
        plt.imshow(x_fusion, cmap='viridis')
        plt.axis('off')
        plt.colorbar()

    plt.ioff()  # turn off interactive mode
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
