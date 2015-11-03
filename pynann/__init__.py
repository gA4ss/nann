#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'devilogic'

'''
Nanan rtificial neural network algorithm library.
'''

import os, re
import nann

'''
print nann version
'''
def version():
    return nann.version()


'''
test nann
'''
def test():
    nann.test()

'''
load nann
'''
def load():
    return nann.load()
    
'''
unload nann
'''
def unload():
    return nann.unload()

'''
create nann manager
'''
def create(task, json, max_calc, now_calc):
    return nann.create(task, json, max_calc, now_calc)

'''
destroy nann manager
'''
def destroy(task):
    
    