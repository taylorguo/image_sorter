# !/usr/bin/env python
# ! coding=utf-8

import os

def get_files_name(folder_name):
    for root, dirs, files in os.walk(folder_name):
        # print('root_dir:', root)  # 当前目录路径
        # print('sub_dirs:', dirs)  # 当前路径下所有子目录
        # print('files:', files)  # 当前路径下所有非目录子文件
        return files

def rename(folder_name):

    files = get_files_name(folder_name)
    print(files)

    i=1
    j=0
    for file in files:

        middle_name = "%03d"%i

        if file.endswith(".jpeg"):
            ext_name = ".jpeg"
            new_file_name = folder_name + "_" + middle_name + ext_name
            i+=1
        elif file.endswith(".jpg"):
            ext_name = ".jpg"
            new_file_name = folder_name + "_" + middle_name + ext_name
            i+=1
        elif file.endswith(".png"):
            ext_name = ".png"
            new_file_name = folder_name + "_" + middle_name + ext_name
            i+=1
        else:
            j+=1
            new_file_name = "no_new_file"

        print("\t", file, new_file_name)
        if not file.startswith("."):
            os.system("mv ./%s/%s ./%s/%s"%(folder_name,file,folder_name,new_file_name))

    if j==0:return i
    else: return j




if __name__ == "__main__":

    target_folder = input("输入包含待更名图片的目录: ")
    # import sys
    # target_folder = sys.argv[1]

    i = rename(target_folder)
    if i > 0:
        print("\t%d images not rename .jpg/.jpeg"%i)
    else:
        print("\t %d finished rename !"%i)