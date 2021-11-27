import os

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
            f.write('\n')
            
for file in os.listdir("filelists"):
    print(file)
    if("ljs" in file):
        f_list = read_list(os.path.join("filelists",file))
        for i in range(len(f_list)):
            f_list[i] = f_list[i].replace("~","/home/v-haoheliu")
        write_list(f_list,os.path.join("filelists",file))