#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'devilogic'

'''
Nanan rtificial neural object.
'''

import pynann

'''
异常类型
'''
nann_config_exptype = 1;
nann_config_already_load = False;

'''
Nann object
'''
class nannobj(object):
    '''类初始化'''
    def __init__(self, task, json, is_file=False, max_calc=1024, curr_calc=100, exptype=1):
        if (nann_config_already_load == False):
            nann_config_already_load = True
            pynann.nann.load()
            
        self.__task = task
        ret = pynann.create(task, json, is_file, max_calc, curr_calc)
        if (ret != 0):
            pass
    
    '''训练'''
    def training(self, json, is_file=False):
        pynann.nann.training(self.__task, json, is_file)
    
    '''获取任务名'''
    def taskname():
        return __task
    
    '''目前还有多少计算结点正在处于计算状态'''
    def __len__(self):
        return pynann.nann.iscalcing(self.__task)
    
    