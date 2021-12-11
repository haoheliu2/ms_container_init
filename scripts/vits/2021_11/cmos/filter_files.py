#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   filter_files.py    
@Contact :   haoheliu@gmail.com
@License :   (C)Copyright 2020-2100

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
11/20/21 8:54 PM   Haohe Liu      1.0         None
'''

import os

CMOS_UTTERRANCE=20

# Model total output
# source_path = "/Users/liuhaohe/Downloads/T3"
source_path = "/Users/liuhaohe/Downloads/output_3617000"
# source_path = "/Users/liuhaohe/OneDrive - Microsoft/CMOS/VITS/GroundTruth"
# Filtered output path
target_path = "/Users/liuhaohe/OneDrive - Microsoft/CMOS/exps/ljs_exp16_reverse"
# target_path = "/Users/liuhaohe/OneDrive - Microsoft/CMOS/VITS/20-2021-11-25-ljs_exp14/baseline_116000"
# target_path = "/Users/liuhaohe/OneDrive - Microsoft/CMOS/VITS/20-2021-12-01-gt-z/baseline_gt"
# Do not need to change
filenames_path = "/Users/liuhaohe/OneDrive - Microsoft/CMOS/VITS/FileNames"

source_files = os.listdir(source_path)

def find_file(fname):
    global source_files
    for each in source_files:
        if(fname in each): return each
    raise ValueError("Error: %s file not found" % (fname))

with open(os.path.join(filenames_path,str(CMOS_UTTERRANCE)+".txt")) as f:
    for i, line in enumerate(f.readlines()):
        line = line.strip()
        src_fname = os.path.join(source_path, find_file(line))
        cmd = "cp \'%s\' \'%s\'" % (src_fname, os.path.join(target_path, "%04d.wav" % (i+1)))
        print(cmd)
        os.system(cmd)

