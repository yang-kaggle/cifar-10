import os
import pandas as pd
import torch
import torchvision
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from utils import *


if __name__ == '__main__':
    # data preprocess
    data_dir = './data/kaggle_cifar10_tiny'
    valid_ratio = 0.1
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)

    # imag augmentation
    transform_train = torchvision.transforms.Compose([
        # 在高度和宽度上将图像放大到40像素的正方形
        torchvision.transforms.Resize(40),
        # 随机裁剪出一个高度和宽度均为40像素的正方形图像，
        # 生成一个面积为原始图像面积0.64到1倍的小正方形，
        # 然后将其缩放为高度和宽度均为32像素的正方形
        torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        # 标准化图像的每个通道
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    # net, parameter, device, optimizer, loss
    batch_size, num_epochs = 128, 1
    lr, wd = 1e-2, 5e-4
    lr_period, lr_decay = 4, 0.9
    param_group = True
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())] \
        if torch.cuda.is_available() else [torch.device('cpu')]
    print('training on', devices[0])

    net = torchvision.models.resnet18(num_classes=10)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0]).to(devices[0])
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_period, lr_decay)

    # choose the best model
    train_ds = ImageFolder(os.path.join(data_dir, 'train_valid_test', 'train'), transform=transform_train)
    valid_ds = ImageFolder(os.path.join(data_dir, 'train_valid_test', 'valid'), transform=transform_test)

    train_iter = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True)
    valid_iter = DataLoader(valid_ds, batch_size, shuffle=False, drop_last=True)

    for epoch in range(num_epochs):
        net.train()
        metric = Accumulator(3)
        for X, y in tqdm(train_iter):
            optimizer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            ls = loss(net(X), y)
            ls.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(ls * X.shape[0], accuracy(net(X), y), X.shape[0])
        scheduler.step()
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        valid_acc = evaluate_accuracy(net, valid_iter)
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, valid acc {valid_acc:.3f}')

    # train the model
    train_ds = ImageFolder(os.path.join(data_dir, 'train_valid_test', 'train_valid'), transform=transform_train)
    test_ds = ImageFolder(os.path.join(data_dir, 'train_valid_test', 'test'), transform=transform_test)

    train_iter = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True)
    test_iter = DataLoader(test_ds, batch_size, shuffle=False, drop_last=False)

    for epoch in range(num_epochs):
        net.train()
        metric = Accumulator(3)
        for X, y in tqdm(train_iter):
            optimizer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            ls = loss(net(X), y)
            ls.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(ls * X.shape[0], accuracy(net(X), y), X.shape[0])
        scheduler.step()
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}')
    torch.save(net.state_dict(), './models/resnet18.pth')

    # predict
    model = net
    model.load_state_dict(torch.load('./models/resnet18.pth', map_location=devices[0]))

    preds = []
    for X, _ in test_iter:
        with torch.no_grad():
            pred_batch = model(X.to(devices[0])).argmax(axis=1)
        preds.extend(pred_batch.type(torch.int32).cpu().numpy())
    sorted_ids = list(range(1, len(test_ds) + 1)).sort(key=lambda x: str(x))
    df = pd.DataFrame({'id': sorted_ids, 'label': preds})
    df['label'] = df['label'].apply(lambda x: train_ds.classes[x])  # 翻译为文字标签
    df.to_csv('submission.csv', index=False)



