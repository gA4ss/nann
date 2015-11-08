#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'devilogic'

'''
Nanan rtificial neural network algorithm library.
'''

import os, re
import nann

'''
print nann version.
'''
def version():
    return nann.version()


'''
test nann.
'''
def test():
    nann.test()

'''
load nann.
'''
def load():
    return nann.load()
    
'''
unload nann.
'''
def unload():
    return nann.unload()

'''
create nann manager.
task : 任务名
json : json缓存 或者 json文件路径
if_file : json参数是否是文件路径
max_calc<可选> : 最大计算结点数，默认1024个
now_calc<可选> : 直接启动线程，默认启动100个
'''
def create(task, json, is_file=False, max_calc=1024, now_calc=100):
    if (is_file == True):
        file_object = open(json,'r')
        try:
            json_text = file_object.read()
        finally:
            file_object.close()
    else:
        json_text = json
    
    return nann.create(task, json_text, max_calc, now_calc)

'''
destroy nann manager.
'''
def destroy(task):
    return nann.destroy(task)
    
    
'''
change except type.
'''
def change_except_type(t):
    return nann.exptype(t)

'''
train sample, adjust weight & output result.
'''
def training(task, json, is_file=False):
    if (is_file == True):
        file_object = open(json,'r')
        try:
            json_text = file_object.read()
        finally:
            file_object.close()
    else:
        json_text = json    
    return nann.training(task, json_text)

'''
train sample, not adjust weight & output result.
'''
def training_notarget(task, json, is_file=False):
    if (is_file == True):
        file_object = open(json,'r')
        try:
            json_text = file_object.read()
        finally:
            file_object.close()
    else:
        json_text = json    
    return nann.training_notarget(task, json_text)

'''
train sample, adjust weight & not output result.
'''
def training_nooutput(task, json, is_file=False):
    if (is_file == True):
        file_object = open(json,'r')
        try:
            json_text = file_object.read()
        finally:
            file_object.close()
    else:
        json_text = json    
    return nann.training_nooutput(task, json_text)

'''
read an ann from nnn file.
'''
def read(filepath):
    return nann.nnn_read(filepath)

'''
write an ann to nnn file.
'''
def write(filepath):
    return nann.nnn_write(filepath)

'''
print ann from task.
'''
def print_info(task):
    nann.print_info(task)

'''
some task is running.
'''
def iscalcing():
    return nann.iscalcing()

'''
merge task's all ann to one.
'''
def merge(task):
    return nann.merge(task)