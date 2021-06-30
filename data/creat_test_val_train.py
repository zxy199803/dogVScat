import os
from sklearn.model_selection import train_test_split
from shutil import copyfile
from utils import write_csv

# 从原始数据中划分训练集，测试集，验证集


def rename_rawdata():
    # 对原始数据集图片文件名增加类别标签 0.jpg -> cat0.jpg

    root = './rawdata/kagglecatsanddogs_3367a/PetImages'
    cat_path = root + '/Cat/'
    dog_path = root + '/Dog/'
    cat_path_list = os.listdir(root + '/Cat')  # 文件夹下图片名列表
    dog_path_list = os.listdir(root + '/Dog')

    for old_name in cat_path_list:
        new_name = root + '/Cat/' + 'cat' + old_name
        os.rename(cat_path + old_name, new_name)

    for old_name in dog_path_list:
        new_name = root + '/Dog/' + 'dog' + old_name
        os.rename(dog_path + old_name, new_name)


def shuffle_train_test(root='./rawdata/kagglecatsanddogs_3367a/PetImages'):
    cat_path_list = os.listdir(root + '/Cat')
    dog_path_list = os.listdir(root + '/Dog')

    # 建立训练集、验证集文件夹
    os.makedirs('./train/cat/')
    os.makedirs('./train/dog/')
    os.makedirs('./valid/cat/')
    os.makedirs('./valid/dog/')

    # 划分测试集
    cat_train_val_set, cat_test_set = train_test_split(cat_path_list, test_size=1000, random_state=42)
    dog_train_val_set, dog_test_set = train_test_split(dog_path_list, test_size=1000, random_state=42)
    # 划分训练集、验证集
    cat_train_set, cat_val_set = train_test_split(cat_train_val_set, test_size=1000, random_state=42)
    dog_train_set, dog_val_set = train_test_split(dog_train_val_set, test_size=1000, random_state=42)

    for train in cat_train_set:
        source = root + '/Cat/' + train
        target = './train/cat/' + train
        copyfile(source, target)

    for train in dog_train_set:
        source = root + '/Dog/' + train
        target = './train/dog/' + train
        copyfile(source, target)

    for val in cat_val_set:
        source = root + '/Cat/' + val
        target = './valid/cat/' + val
        copyfile(source, target)

    for val in dog_val_set:
        source = root + '/Dog/' + val
        target = './valid/dog/' + val
        copyfile(source, target)

    # 建立测试集
    os.makedirs('./test/')
    name = 1
    results = []
    for cat in cat_test_set:
        source = root + '/Cat/' + cat
        target = './test/' + str(name) + '.jpg'
        result = [(name, 'cat')]
        results += result
        name += 1
        copyfile(source, target)

    for dog in dog_test_set:
        source = root + '/Dog/' + dog
        target = './test/' + str(name) + '.jpg'
        copyfile(source, target)
        result = [(name, 'dog')]
        results += result
        name += 1

    write_csv(results, './answer.csv')  # 测试集标签


if __name__ == '__main__':
    rename_rawdata()
    shuffle_train_test()
