#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   select_test_files.py    
@Contact :   haoheliu@gmail.com
@License :   (C)Copyright 2020-2100

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
11/20/21 8:53 PM   Haohe Liu      1.0         None
'''


def read_list(fname):
    result = []
    with open(fname, "r") as f:
        for each in f.readlines():
            each = each.strip('\n')
            result.append(each)
    return result

def write_list(list, fname):
    with open(fname,'w') as f:
        for word in list:
            f.write(word)
            # f.write('\n')


import os

root_path = "/Users/liuhaohe/Listener_test/vits/LJSpeech-1.1/wavs"
files = "/Users/liuhaohe/OneDrive - Microsoft/CMOS/VITS/FileNames/500.txt"

files = read_list(files)
for file in files:
    path = os.path.join(root_path, file)
    cmd = "cp %s \'%s\' " % (path, "/Users/liuhaohe/OneDrive - Microsoft/CMOS/VITS/GroundTruth")
    os.system(cmd)
    # print(cmd)