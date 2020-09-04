# -*- coding:gb2312 -*-
# -*- coding:UTF-8 -*-
# @Time     :2020 09 2020/9/3 11:10
# @Author   :千乘
import os
import models
from config import DefaultConfig
from data import dataset
from utils import visualize
from torch.utils.data import DataLoader
import torch as t
from torch.autograd import Variable
from torchnet import meter
import pandas as pd

opt = DefaultConfig()


def train(**kwargs):
    # 配置更新
    opt.parse(kwargs)
    vis = visualize.Visualizer(opt.env)

    # 模型
    model = getattr(models, opt.model)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # 数据
    train_data = dataset.DogCat(opt.train_data_root, train=True)
    val_data = dataset.DogCat(opt.train_data_root, train=False)
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(train_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # 目标函数，优化器
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)

    # 统计指标
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    # 训练
    for epoch in range(opt.max_epoch):

        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, label) in enumerate(train_dataloader):

            # 训练模型
            input = Variable(data)
            target = Variable(label)
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            # 更新统计指标以及可视化
            loss_meter.add(loss.data[0])
            confusion_matrix.add(score.data, target.data)

            if ii % opt.print_freq == opt.print_freq - 1:
                vis.plot('loss', loss_meter.value()[0])

                # 如果需要的话，进入debug模式
                if os.path.exists(opt.debug_file):
                    import ipdb
                    ipdb.set_trace()

        model.save()

        # 计算验证集上的指标及可视化
        val_cm, val_accuracy = val(model, val_dataloader)
        vis.plot('val_accuracy', val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}"
            .format(
            epoch=epoch,
            loss=loss_meter.value()[0],
            val_cm=str(val_cm.value()),
            train_cm=str(confusion_matrix.value()),
            lr=lr))

        # 如果损失不再下降，则降低学习率
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]


def val(model, dataloader):
    '''
    计算模型在验证集上的准确率等信息
    '''
    # 把模型设为验证模式
    model.eval()

    confusion_matrix = meter.ConfusionMeter(2)
    for ii, data in enumerate(dataloader):
        input, label = data
        val_input = Variable(input, volatile=True)
        val_label = Variable(label.long(), volatile=True)
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        score = model(val_input)
        confusion_matrix.add(score.data.squeeze(), label.long())

    # 把模型恢复为训练模式
    model.train()

    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / \
               (cm_value.sum())
    return confusion_matrix, accuracy


def test(**kwargs):
    opt.parse(kwargs)
    # 模型
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

    # 数据
    train_data = dataset.DogCat(opt.test_data_root, test=True)
    test_dataloader = DataLoader(train_data,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.num_workers)

    results = []
    for ii, (data, path) in enumerate(test_dataloader):
        input = t.autograd.Variable(data, volatile=True)
        if opt.use_gpu: input = input.cuda()
        score = model(input)
        probability = t.nn.functional.softmax(score)[:, 1].data.tolist()
        batch_results = [(path_, probability_) for path_, probability_ in zip(path, probability)]
        results += batch_results
    results = pd.DataFrame(results)
    results.to_csv(opt.result_file, header=True, index=True)
    return results


def help():
    print('help')
    '''
    打印帮助的信息： python file.py help
     '''

    print('''
     usage : python {0} <function> [--args=value,]
     <function> := train | test | help
     example: 
             python {0} train --env='env0701' --lr=0.01
             python {0} test --dataset='path/to/dataset/root/'
             python {0} help
     avaiable args:'''.format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':
    import fire

    fire.Fire()
