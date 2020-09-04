# -*- coding:gb2312 -*-
# -*- coding:UTF-8 -*-
# @Time     :2020 09 2020/9/3 11:09
# @Author   :千乘

import torch as t
import time

'''简易封装Module,提供接口'''


class BasicModule(t.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        '''
        加载指定路径模型
        :param path: path
        :return: None
        '''
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        '''
        保存模型，使用'模型名字+时间'作为文件名
        :return: name

        '''
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name
