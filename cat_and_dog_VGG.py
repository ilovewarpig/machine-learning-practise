'''
CNN模型，使用预训练后的模型，可用于在小数据集上训练高质量模型
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
import os
import re
import random


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
        # gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        # resized_gray_img = cv2.resize(gray_img, (100, 100))
        resized_rgb_img = cv2.resize(cropped_img, (150, 150))
        # black_whites.append(resized_gray_img)
        black_whites.append(resized_rgb_img)
    return np.asarray(black_whites)


if __name__ == '__main__':
    test_dir = []
    train_dir = []
    test_x = []
    train_x = []
    test_y = []
    train_y = []

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

    train_x = crop_pics(train_dir)
    validation_x = crop_pics(validation_dir)
    test_x = crop_pics(test_dir)

    reshaped_x_test = test_x.reshape(-1, 150, 150, 3)
    reshaped_x_validation = validation_x.reshape(-1, 150, 150, 3)
    reshaped_x_train = train_x.reshape(-1, 150, 150, 3)

    # 数据增强
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow(reshaped_x_train, train_y, batch_size=32)
    validation_generator = train_datagen.flow(reshaped_x_validation, validation_y, batch_size=32)
    test_generator = test_datagen.flow(reshaped_x_test, test_y, batch_size=32)

    # 获取VGG16
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    conv_base.trainable = False
    model = Sequential()
    model.add(conv_base)
    # 添加分类器
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit_generator(train_generator, steps_per_epoch=187, epochs=30,
                                   validation_data=validation_generator, validation_steps=62)

    test_loss, test_acc = model.evaluate(reshaped_x_test, test_y)
    print(test_acc)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    plt.figure()
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['Accuracy', 'Validation Accuracy'])
    plt.show()
