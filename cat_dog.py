'''
简单CNN模型，不使用genaretor
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
import os
import re
import random


# 数据增强
def data_augment(sets, train_x, train_y):
    for pics in sets:
        train_x.append(cv2.imread(pics))
    for item in sets:
        # 原生
        img_original = cv2.imread(item)
        y = train_y[sets.index(item)]
        # train_x.append(img_original)
        # train_y.append(y)
        # 水平翻转
        img_fliped = cv2.flip(img_original, 1)
        train_x.append(img_fliped)
        train_y.append(y)
        # 缩放
        img_resized = cv2.resize(img_original, (int(img_original.shape[1] * random.uniform(0.6, 1.4)),\
                                                int(img_original.shape[0] * random.uniform(0.6, 1.4))))
        train_x.append(img_resized)
        train_y.append(y)
        # 旋转(0~180)
        M = cv2.getRotationMatrix2D((int(img_original.shape[1]/2), int(img_original.shape[0]/2)), random.randint(1, 180), 1)
        img_rotation = cv2.warpAffine(img_original, M, (img_original.shape[1], img_original.shape[0]))
        train_x.append(img_rotation)
        train_y.append(y)
    return (train_x, train_y)


# 训练集灰度截图
def train_crop_pics(sets):
    black_whites = []
    for item in sets:
        img = item
        img_height = img.shape[0]
        img_width = img.shape[1]
        if img_width > img_height:
            col_start = int((img_width - img_height) / 2)
            col_end = col_start + img_width
            cropped_img = img[:, col_start:col_end, :]
        else:
            row_start = int((img_height - img_width) / 2)
            row_end = row_start + img_height
            cropped_img = img[row_start:row_end, :, :]
        gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        resized_gray_img = cv2.resize(gray_img, (100, 100))
        black_whites.append(resized_gray_img)
    return np.asarray(black_whites)


# 灰度截图
def crop_pics(sets):
    black_whites = []
    for item in sets:
        img = cv2.imread(item)
        img_height = img.shape[0]
        img_width = img.shape[1]
        if img_width > img_height:
            col_start = int((img_width - img_height) / 2)
            col_end = col_start + img_width
            cropped_img = img[:, col_start:col_end, :]
        else:
            row_start = int((img_height - img_width) / 2)
            row_end = row_start + img_height
            cropped_img = img[row_start:row_end, :, :]
        gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        resized_gray_img = cv2.resize(gray_img, (100, 100))
        black_whites.append(resized_gray_img)

    return np.asarray(black_whites)


# rgb截图
def crop_pics_rgb(sets):
    black_whites2 = []
    for item in sets:
        img = cv2.imread(item)
        img_height = img.shape[0]
        img_width = img.shape[1]
        if img_width > img_height:
            col_start = int((img_width - img_height) / 2)
            col_end = col_start + img_width
            cropped_img = img[:, col_start:col_end, :]
        else:
            row_start = int((img_height - img_width) / 2)
            row_end = row_start + img_height
            cropped_img = img[row_start:row_end, :, :]
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        resized_rgb_img = cv2.resize(cropped_img, (100, 100))
        black_whites2.append(resized_rgb_img)
    return np.asarray(black_whites2)


def test(model):
    test_dir = []
    labels = []
    for i in os.listdir('C:\\Users\\Leo-Li\\Desktop\\test_pics'):
        if '_' not in i:
            test_dir.append('C:\\Users\\Leo-Li\\Desktop\\test_pics\\' + i)
    mytest = crop_pics(test_dir[:12])
    pics = crop_pics_rgb(test_dir[:12])
    mytest = tf.keras.utils.normalize(mytest, axis=1)
    reshaped_mytest = mytest.reshape(-1, 100, 100, 1)

    predictions = model2.predict(reshaped_mytest)
    for item in predictions:
        if item[0] > 0.5:
            print('cat')
            labels.append('cat')
        else:
            print('dog')
            labels.append('dog')
    for i in range(len(labels)):
        plt.subplot(3, 4, i + 1)
        plt.imshow(pics[i])
        plt.text(0, 20, labels[i], fontsize=16, color='r')


if __name__ == '__main__':
    test_dir = []
    train_dir = []
    test_x = []
    train_x = []
    test_y = []
    train_y = []
    validation_x = []
    # 读取图片文件路径
    for i in os.listdir('C:\\Users\\Leo-Li\\Downloads\\cat-and-dog\\test_set\\cats'):
        if '_' not in i:
            test_dir.append('C:\\Users\\Leo-Li\\Downloads\\cat-and-dog\\test_set\\cats\\' + i)

    for i in os.listdir('C:\\Users\\Leo-Li\\Downloads\\cat-and-dog\\test_set\\dogs'):
        if '_' not in i:
            test_dir.append('C:\\Users\\Leo-Li\\Downloads\\cat-and-dog\\test_set\\dogs\\' + i)

    for i in os.listdir('C:\\Users\\Leo-Li\\Downloads\\cat-and-dog\\training_set\\cats'):
        if '_' not in i:
            train_dir.append('C:\\Users\\Leo-Li\\Downloads\\cat-and-dog\\training_set\\cats\\' + i)

    for i in os.listdir('C:\\Users\\Leo-Li\\Downloads\\cat-and-dog\\training_set\\dogs'):
        if '_' not in i:
            train_dir.append('C:\\Users\\Leo-Li\\Downloads\\cat-and-dog\\training_set\\dogs\\' + i)

    # 随机打乱
    random.shuffle(train_dir)
    random.shuffle(test_dir)
    # 根据训练集和测试集路径生成相应的标识
    for item in train_dir:
        if re.findall('set\\\(.*?)\\\\', item, re.S)[0] == 'cats':
            train_y.append(1)
        else:
            train_y.append(0)

    for item in test_dir:
        if re.findall('set\\\(.*?)\\\\', item, re.S)[0] == 'cats':
            test_y.append(1)
        else:
            test_y.append(0)

    # 划分训练集和验证集
    validation_dir = train_dir[int(len(train_dir) * 0.75):len(train_dir)]
    validation_y = train_y[int(len(train_y) * 0.75):len(train_y)]
    train_dir = train_dir[:int(len(train_dir) * 0.75)]
    train_y = train_y[:int(len(train_y) * 0.75)]

    # 灰度截图
    (train_x, train_y) = data_augment(train_dir, train_x, train_y)
    train_x = train_crop_pics(train_x)
    validation_x = crop_pics(validation_dir)
    test_x = crop_pics(test_dir)

    # train_x = crop_pics_rgb(train_x)
    # test_x = crop_pics_rgb(test_x)

    # 再次打乱训练集
    state = np.random.get_state()
    np.random.shuffle(train_x)
    np.random.set_state(state)
    np.random.shuffle(train_y)

    # 标准化
    train_x = tf.keras.utils.normalize(train_x, axis=1)
    validation_x = tf.keras.utils.normalize(validation_x, axis=1)
    test_x = tf.keras.utils.normalize(test_x, axis=1)
    # 重塑
    reshaped_x_test = test_x.reshape(-1, 100, 100, 1)
    reshaped_x_validation = validation_x.reshape(-1, 100, 100, 1)
    reshaped_x_train = train_x.reshape(-1, 100, 100, 1)

    # 建立模型
    model2 = Sequential()
    model2.add(Conv2D(64, (3, 3), activation='relu', input_shape=(100, 100, 1)))
    model2.add(MaxPool2D(pool_size=(2, 2)))
    # model2.add(tf.keras.layers.BatchNormalization())
    model2.add(Conv2D(64, (3, 3), activation='relu'))
    model2.add(MaxPool2D(pool_size=(2, 2)))
    # model2.add(tf.keras.layers.BatchNormalization())
    model2.add(Conv2D(64, (3, 3), activation='relu'))
    model2.add(MaxPool2D(pool_size=(2, 2)))
    model2.add(Flatten())
    model2.add(Dropout(0.5))
    model2.add(Dense(64, activation='relu'))
    model2.add(Dense(1, activation='sigmoid'))
    model2.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])

    # 训练模型
    train_result2 = model2.fit(reshaped_x_train, train_y, epochs=15, validation_data=(reshaped_x_validation, validation_y))

    # 保存模型
    model2.save('cat_and_dogs_a_20_5.h5')
    # cat_and_dogs_a_20 训练了10次 80% 拟合情况挺好
    # mode2.save('cat_and_dogs_a.h5') 85% v_acc

    test_loss, test_acc = model2.evaluate(reshaped_x_test, test_y)
    print(test_acc)
    # 检查模型精确度
    acc = train_result2.history['acc']
    val_acc = train_result2.history['val_acc']
    plt.figure()
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['Accuracy', 'Validation Accuracy'])
    plt.show()
