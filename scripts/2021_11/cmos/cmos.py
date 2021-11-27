#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   cmos.py    
@Contact :   haoheliu@gmail.com
@License :   (C)Copyright 2020-2100

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
11/20/21 8:53 PM   Haohe Liu      1.0         None
'''

 

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   temp.py
@Contact :   haoheliu@gmail.com
@License :   (C)Copyright 2020-2100

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
11/20/21 8:30 PM   Haohe Liu      1.0         None
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
file = "filelists/ljs_audio_text_test_filelist.txt"

for i in range(10,520,10):
    with open(file, "r") as f:
        lines = f.readlines()
        for j in range(len(lines)):
            lines[j] = lines[j].split("|")[1]
    write_list(lines[:i], os.path.join("/Users/liuhaohe/OneDrive - Microsoft/CMOS/VITS/GroundSentences",str(i)+".txt"))


