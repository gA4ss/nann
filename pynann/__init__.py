#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'devilogic'

'''
Nanan rtificial neural network algorithm library.
'''

import os, re
import nann

#----------------------------------------------------------------------
def version():
    """print nann version."""
    return nann.version()

#----------------------------------------------------------------------
def test():
    """test nann."""
    nann.test()
    
#----------------------------------------------------------------------
def change_except_type(t):
    """t != 0 throw except, t == 0 return errcode"""
    return nann.exptype(t)

#----------------------------------------------------------------------
def training(task, ann_json, input_json, wt=0, is_file=False):
    """train sample, adjust weight & output result."""
    if (is_file == True):
        ann_file_object = open(ann_json,'r')
        input_file_object = open(input_json, 'r')
        try:
            ann_json_text = ann_file_object.read()
            input_json_text = input_file_object.read()
        finally:
            ann_file_object.close
            input_file_object.close
    else:
        ann_json_text = ann_json
        input_json_text = input_json
    return nann.training(task, ann_json_text, input_json_text, wt)

#----------------------------------------------------------------------
def done(task):
    """check task is done"""
    return nann.done(task)
    
#----------------------------------------------------------------------
def clear(task):
    """clear task which is done."""
    nann.clear(task)
    
#----------------------------------------------------------------------
def clears():
    """clear all tasks which is done."""
    nann.clears()
    
#----------------------------------------------------------------------
def get_map_results(task):
    """get task map results."""
    return nann.get_map_results(task)
    
#----------------------------------------------------------------------
def get_reduce_result(task):
    """get task reduce result."""
    return nann.get_reduce_result(task)
    
#----------------------------------------------------------------------
def waits():
    """wait all task done."""
    nann.waits()
    
#----------------------------------------------------------------------
def wait(task):
    """wait task done."""
    nann.wait(task)    
    
#----------------------------------------------------------------------
def set_precision(precison=4):
    """set ann output precision."""
    nann.set_precision(precison)
    
#----------------------------------------------------------------------
def start_auto_clear():
    """start auto clear thread"""
    nann.start_auto_clear()
    
#----------------------------------------------------------------------
def stop_auto_clear():
    """stop auto clear thread"""
    nann.stop_auto_clear()
    
    

