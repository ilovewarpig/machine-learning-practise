import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
import os
import re
import random
from keras.utils.np_utils import  to_categorical


tf.disable_v2_behavior()


# 截取单个彩色图片 用于视频检测
def crop_single_pic_rgb(img):
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
    resized_rgb_img = cv2.resize(cropped_img, (150, 150))

    return np.asarray(resized_rgb_img)


# 截取灰度图片
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
        gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGBA2GRAY)
        resized_gray_img = cv2.resize(gray_img, (150, 150))
        black_whites.append(resized_gray_img)

    return np.asarray(black_whites)


# 截取RGB图片
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
        resized_rgb_img = cv2.resize(cropped_img, (150, 150))
        black_whites2.append(resized_rgb_img)
    return np.asarray(black_whites2)


# 测试函数，输出3行4列的图片并写上预测结果
def test(model):
    test_dir = []
    for i in os.listdir('C:\\Users\\ilovewarpig\\Desktop\\test_pics'):
            test_dir.append('C:\\Users\\ilovewarpig\\Desktop\\test_pics\\' + i)
    mytest = crop_pics_rgb(test_dir)

    reshaped_mytest = mytest.reshape(-1, 150, 150, 3)
    mytest_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    mytest_datagen.fit(reshaped_mytest)
    mytest_generator = mytest_datagen.flow(reshaped_mytest, shuffle=False)

    predictions = model.predict_generator(mytest_generator)
    yhat = np.argmax(predictions, axis=1)
    print(yhat)

    dic = {0: 'breadboard', 1: 'buzzer', 2: 'frame', 3: 'hoare', 4: 'joystick', 5: 'mainchip', 6: 'motor', 7: 'relay',
           8: 'steering', 9: 'ultrasonic'}
    labels = list(map(dic.get, yhat))
    print(labels)
    for pic in range(len(labels)):
        plt.subplot(3, 4, pic + 1)
        plt.imshow(mytest[pic])
        plt.text(0, 20, labels[pic], fontsize=16, color='r')
    plt.show()


# 对摄像头拍摄的画面进行测试
def video_test(model):
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('camera')
    font = cv2.FONT_HERSHEY_DUPLEX
    count = 0
    while cap.isOpened():
        print('count')
        count += 1
        ret, frame = cap.read()
        if ret:
            predict = crop_single_pic_rgb(frame)
            reshaped_predict = predict.reshape(-1, 150, 150, 3)
            predict_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
            predict_datagen.fit(reshaped_predict)
            predict_generator = predict_datagen.flow(reshaped_predict, shuffle=False)

            predictions = model.predict_generator(predict_generator)
            yhat = np.argmax(predictions, axis=1)
            dic = {0: 'breadboard', 1: 'buzzer', 2: 'flame', 3: 'hoare', 4: 'joystick', 5: 'mainchip', 6: 'motor',
                   7: 'relay', 8: 'steering', 9: 'ultrasonic'}
            labels = list(map(dic.get, yhat))

            img_font = cv2.putText(frame, labels[0], (150, 100), font, 2, (0, 0, 0), 2, )
            cv2.imshow('camera', frame)
            cv2.waitKey(1)
        if count >= 1000:
            break
    print('TIME OUT')
    return


if __name__ == '__main__':
    test_dir = []
    train_dir = []
    test_x = []
    train_x = []
    test_y = []
    train_y = []

    # 读取图片文件路径
    item_list = ['breadboard', 'buzzer', 'fire', 'hoare', 'joystick', 'mainchip', 'motor', 'relay', 'steering', 'ultrasonic']
    for item in item_list:
        for i in os.listdir('C:\\Users\\ilovewarpig\\Desktop\\test\\{}_frames_highQ'.format(item)):
            print('C:\\Users\\ilovewarpig\\Desktop\\test\\{}_frames_highQ\\'.format(item) + i)

    for item in item_list:
        for i in os.listdir('C:\\Users\\ilovewarpig\\Desktop\\train\\{}_frames_highQ'.format(item)):
            print('C:\\Users\\ilovewarpig\\Desktop\\train\\{}_frames_highQ\\'.format(item) + i)
    '''        
    for i in os.listdir('C:\\Users\\ilovewarpig\\Desktop\\test\\breadboard_frames_highQ'):
        test_dir.append('C:\\Users\\ilovewarpig\\Desktop\\test\\breadboard_frames_highQ\\' + i)

    for i in os.listdir('C:\\Users\\ilovewarpig\\Desktop\\test\\buzzer_frames_highQ'):
        test_dir.append('C:\\Users\\ilovewarpig\\Desktop\\test\\buzzer_frames_highQ\\' + i)

    for i in os.listdir('C:\\Users\\ilovewarpig\\Desktop\\test\\fire_frames_highQ'):
        test_dir.append('C:\\Users\\ilovewarpig\\Desktop\\test\\fire_frames_highQ\\' + i)

    for i in os.listdir('C:\\Users\\ilovewarpig\\Desktop\\test\\hoare_frames_highQ'):
        test_dir.append('C:\\Users\\ilovewarpig\\Desktop\\test\\hoare_frames_highQ\\' + i)

    for i in os.listdir('C:\\Users\\ilovewarpig\\Desktop\\test\\joystick_frames_highQ'):
        test_dir.append('C:\\Users\\ilovewarpig\\Desktop\\test\\joystick_frames_highQ\\' + i)

    for i in os.listdir('C:\\Users\\ilovewarpig\\Desktop\\test\\mainchip_frames_highQ'):
        test_dir.append('C:\\Users\\ilovewarpig\\Desktop\\test\\mainchip_frames_highQ\\' + i)

    for i in os.listdir('C:\\Users\\ilovewarpig\\Desktop\\test\\motor_frames_highQ'):
        test_dir.append('C:\\Users\\ilovewarpig\\Desktop\\test\\motor_frames_highQ\\' + i)

    for i in os.listdir('C:\\Users\\ilovewarpig\\Desktop\\test\\relay_frames_highQ'):
        test_dir.append('C:\\Users\\ilovewarpig\\Desktop\\test\\relay_frames_highQ\\' + i)

    for i in os.listdir('C:\\Users\\ilovewarpig\\Desktop\\test\\steering_frames_highQ'):
        test_dir.append('C:\\Users\\ilovewarpig\\Desktop\\test\\steering_frames_highQ\\' + i)

    for i in os.listdir('C:\\Users\\ilovewarpig\\Desktop\\test\\ultrasonic_frames_highQ'):
        test_dir.append('C:\\Users\\ilovewarpig\\Desktop\\test\\ultrasonic_frames_highQ\\' + i)

    for i in os.listdir('C:\\Users\\ilovewarpig\\Desktop\\train\\breadboard_frames_highQ'):
        train_dir.append('C:\\Users\\ilovewarpig\\Desktop\\train\\breadboard_frames_highQ\\' + i)

    for i in os.listdir('C:\\Users\\ilovewarpig\\Desktop\\train\\buzzer_frames_highQ'):
        train_dir.append('C:\\Users\\ilovewarpig\\Desktop\\train\\buzzer_frames_highQ\\' + i)

    for i in os.listdir('C:\\Users\\ilovewarpig\\Desktop\\train\\fire_frames_highQ'):
        train_dir.append('C:\\Users\\ilovewarpig\\Desktop\\train\\fire_frames_highQ\\' + i)

    for i in os.listdir('C:\\Users\\ilovewarpig\\Desktop\\train\\hoare_frames_highQ'):
        train_dir.append('C:\\Users\\ilovewarpig\\Desktop\\train\\hoare_frames_highQ\\' + i)

    for i in os.listdir('C:\\Users\\ilovewarpig\\Desktop\\train\\joystick_frames_highQ'):
        train_dir.append('C:\\Users\\ilovewarpig\\Desktop\\train\\joystick_frames_highQ\\' + i)

    for i in os.listdir('C:\\Users\\ilovewarpig\\Desktop\\train\\mainchip_frames_highQ'):
        train_dir.append('C:\\Users\\ilovewarpig\\Desktop\\train\\mainchip_frames_highQ\\' + i)

    for i in os.listdir('C:\\Users\\ilovewarpig\\Desktop\\train\\motor_frames_highQ'):
        train_dir.append('C:\\Users\\ilovewarpig\\Desktop\\train\\motor_frames_highQ\\' + i)

    for i in os.listdir('C:\\Users\\ilovewarpig\\Desktop\\train\\relay_frames_highQ'):
        train_dir.append('C:\\Users\\ilovewarpig\\Desktop\\train\\relay_frames_highQ\\' + i)

    for i in os.listdir('C:\\Users\\ilovewarpig\\Desktop\\train\\steering_frames_highQ'):
        train_dir.append('C:\\Users\\ilovewarpig\\Desktop\\train\\steering_frames_highQ\\' + i)

    for i in os.listdir('C:\\Users\\ilovewarpig\\Desktop\\train\\ultrasonic_frames_highQ'):
        train_dir.append('C:\\Users\\ilovewarpig\\Desktop\\train\\ultrasonic_frames_highQ\\' + i)
    '''
    # 随机打乱
    random.shuffle(train_dir)
    random.shuffle(test_dir)
    # 根据训练集和测试集路径生成相应的标识
    for item in train_dir:
        if re.findall('train\\\\(.*?)_frames_highQ', item, re.S)[0] == 'breadboard':
            train_y.append(0)
        elif re.findall('train\\\\(.*?)_frames_highQ', item, re.S)[0] == 'buzzer':
            train_y.append(1)
        elif re.findall('train\\\\(.*?)_frames_highQ', item, re.S)[0] == 'fire':
            train_y.append(2)
        elif re.findall('train\\\\(.*?)_frames_highQ', item, re.S)[0] == 'hoare':
            train_y.append(3)
        elif re.findall('train\\\\(.*?)_frames_highQ', item, re.S)[0] == 'joystick':
            train_y.append(4)
        elif re.findall('train\\\\(.*?)_frames_highQ', item, re.S)[0] == 'mainchip':
            train_y.append(5)
        elif re.findall('train\\\\(.*?)_frames_highQ', item, re.S)[0] == 'motor':
            train_y.append(6)
        elif re.findall('train\\\\(.*?)_frames_highQ', item, re.S)[0] == 'relay':
            train_y.append(7)
        elif re.findall('train\\\\(.*?)_frames_highQ', item, re.S)[0] == 'steering':
            train_y.append(8)
        else:
            train_y.append(9)

    for item in test_dir:
        if re.findall('test\\\\(.*?)_frames_highQ', item, re.S)[0] == 'breadboard':
            test_y.append(0)
        elif re.findall('test\\\\(.*?)_frames_highQ', item, re.S)[0] == 'buzzer':
            test_y.append(1)
        elif re.findall('test\\\\(.*?)_frames_highQ', item, re.S)[0] == 'fire':
            test_y.append(2)
        elif re.findall('test\\\\(.*?)_frames_highQ', item, re.S)[0] == 'hoare':
            test_y.append(3)
        elif re.findall('test\\\\(.*?)_frames_highQ', item, re.S)[0] == 'joystick':
            test_y.append(4)
        elif re.findall('test\\\\(.*?)_frames_highQ', item, re.S)[0] == 'mainchip':
            test_y.append(5)
        elif re.findall('test\\\\(.*?)_frames_highQ', item, re.S)[0] == 'motor':
            test_y.append(6)
        elif re.findall('test\\\\(.*?)_frames_highQ', item, re.S)[0] == 'relay':
            test_y.append(7)
        elif re.findall('test\\\\(.*?)_frames_highQ', item, re.S)[0] == 'steering':
            test_y.append(8)
        else:
            test_y.append(9)

    # 划分验证机和训练集，比例为1:3
    validation_dir = train_dir[int(len(train_dir) * 0.75):len(train_dir)]
    validation_y = train_y[int(len(train_y) * 0.75):len(train_y)]
    train_dir = train_dir[:int(len(train_dir) * 0.75)]
    train_y = train_y[:int(len(train_y) * 0.75)]
    print('train set:  ',len(train_dir), 'validation set:  ',len(validation_dir), 'test set:  ', len(test_dir))


    # 截取彩色截图，输出中央150*150的图片集
    train_x = crop_pics_rgb(train_dir)
    validation_x = crop_pics_rgb(validation_dir)
    test_x = crop_pics_rgb(test_dir)

    '''
    oneh_hot_train_y = tf.one_hot(train_y, 10).eval(session=tf.Session())
    oneh_hot_test_y = tf.one_hot(test_y, 10).eval(session=tf.Session())
    oneh_hot_validation_y = tf.one_hot(validation_y, 10).eval(session=tf.Session())
    '''
    # 将整数标签转化为onehot编码
    oneh_hot_train_y = to_categorical(train_y)
    oneh_hot_test_y = to_categorical(test_y)
    oneh_hot_validation_y = to_categorical(validation_y)


    # 重塑
    reshaped_x_test = test_x.reshape(-1, 150, 150, 3)
    reshaped_x_validation = validation_x.reshape(-1, 150, 150, 3)
    reshaped_x_train = train_x.reshape(-1, 150, 150, 3)

    # 数据增强（验证集和测试集不要增强）、归一化
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    train_datagen.fit(reshaped_x_train)
    validation_datagen.fit(reshaped_x_validation)
    test_datagen.fit(reshaped_x_test)

    # 创建生成器，每张图生成32张增强图片
    train_generator = train_datagen.flow(reshaped_x_train, oneh_hot_train_y, batch_size=32)
    validation_generator = train_datagen.flow(reshaped_x_validation, oneh_hot_validation_y, batch_size=32)
    test_generator = test_datagen.flow(reshaped_x_test, oneh_hot_test_y, batch_size=32)

    # 建立模型
    model2 = Sequential()
    model2.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model2.add(MaxPool2D(pool_size=(2, 2)))
    # model2.add(tf.keras.layers.BatchNormalization())
    model2.add(Conv2D(64, (3, 3), activation='relu'))
    model2.add(MaxPool2D(pool_size=(2, 2)))
    # model2.add(tf.keras.layers.BatchNormalization())
    model2.add(Conv2D(128, (3, 3), activation='relu'))
    model2.add(MaxPool2D(pool_size=(2, 2)))
    model2.add(Flatten())
    model2.add(Dropout(0.5))
    model2.add(Dense(512, activation='relu'))
    model2.add(Dense(10, activation='softmax'))
    model2.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model2.fit_generator(train_generator, steps_per_epoch=144, epochs=27, validation_data=validation_generator, validation_steps=48)
    model2.save('arduino11.h5')
    # model2 = tf.keras.models.load_model('arduino10.h5')
    # arduino10.h5 acc:92 val_acc:95 test:94

    test_loss, test_acc = model2.evaluate_generator(test_generator)
    print(test_acc)
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    plt.figure()
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['Accuracy', 'Validation Accuracy'])
    plt.show()

    plt.figure()
    i = 0
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(test_x[i])
    print(test_y[:9])

    test(model2)

    video_test(model2)


