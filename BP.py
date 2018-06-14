################################################################################################
                                       # import necessary modules
################################################################################################
import numpy as np
import math
import cv2
import sys
import time
import os
import struct
sys.path.append("..")
from sklearn.metrics import accuracy_score


################################################################################################
                                       # define parameters  #
################################################################################################


feature_len = 2025
out_num = 10  # number of output layer
hid_num = 150  # number of hidden layer ( can be changed )
inp_lrate = 0.1  # learning rate of input layer
hid_lrate = 0.1  # learning rate of hidden layer
test_times = 7 # number of times of training

  ###################################################################################################

                                           # define of functions

  ###################################################################################################

def get_act(x):          # sigmoid function
    act_vec = []
   # print x[:10]
    for i in x:
        if i>100 :
            act_vec.append(1)
        elif i<-100:
            act_vec.append(0)
        else:
            act_vec.append(1 / (1 + math.exp(-i)))
    act_vec = np.array(act_vec)
    return act_vec

def get_suq_err(e):       # Differential equation
    return 0.5 * np.dot(e, e)


def binaryzation(img):
    cv_img = img.astype(np.uint8)
    cv2.threshold(cv_img, 127, 1, cv2.THRESH_BINARY_INV, cv_img)
    return cv_img


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images.idx3-ubyte'% kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


def BPClassify(dataSet, labels, train_time):
    w1 = 0.2 * np.random.random((feature_len, hid_num)) - 0.1  # initialize weight mattrix of input layer
    w2 = 0.2 * np.random.random((hid_num, out_num)) - 0.1  # initialize weight mattrix of hidden layer 1
    hid_offset = np.zeros(hid_num)  # set the offset vector of hidden layer 1
    out_offset = np.zeros(out_num)  # set the offset vector of output layer
    samp_num = len(dataSet)  #total number of sample
    for i in range(0,train_time):
        for count in range(0, samp_num):
            t_label = np.zeros(out_num)
            t_label[labels[count]] = 1
            test_image = binaryzation(dataSet[count])
            #test_image = dataSet[i]
          # test_image = dataSet[count]/256
            # forward process
            hid_value = np.dot(test_image, w1) + hid_offset  
            hid_act = get_act(hid_value) 
            out_value = np.dot(hid_act, w2) + out_offset  
            out_act = get_act(out_value)
             # backward process
            e = t_label - out_act  
            out_delta = e * out_act * (1 - out_act) 
            hid_delta = hid_act * (1 - hid_act) * np.dot(w2, out_delta)  

            for i in range(0, out_num):
                w2[:, i] += hid_lrate * out_delta[i] * hid_act  # update weight vector between hidden layer and output layer
            for i in range(0, hid_num):
                w1[:, i] += inp_lrate * hid_delta[i] * test_image  # update weight vector between hidden layer and input layer
            out_offset += hid_lrate * out_delta  # update offset vector
            hid_offset += inp_lrate * hid_delta
    return w1,w2,hid_offset,out_offset

def BPtest(dataSet,w1,w2,hid_offset,out_offset):
    predict_lables = []
    for i in range(0,len(dataSet)):
        test_image = binaryzation(dataSet[i])
        #test_image = dataSet[i]
        #test_image = dataSet[i] / 256
        hid_value = np.dot(test_image, w1) + hid_offset  # hidden layer value
        hid_act = get_act(hid_value)  # activation value of hidden value
        out_value = np.dot(hid_act, w2) + out_offset  # output layer value
        out_act = get_act(out_value)
        max = out_act[0]
        max_label = 0
        for j in range(0, len(out_act)):
            if out_act[j] > max:
                max = out_act[j]
                max_label = j
        predict_lables.append(max_label)
    #predict_lables = np.array(predict_lables)
    #print (predict_lables[:20])
    return predict_lables




def main():
    print('Reading data...')
    time_1 = time.time()
    train_images = np.fromfile("D:\DataSci_data\mnist\mnist_train_data",dtype=np.uint8)
    train_images = train_images.reshape(60000,2025)
    train_labels = np.fromfile("D:\DataSci_data\mnist\mnist_train_label",dtype=np.uint8)
    train_labels = train_labels.reshape(60000,1)
    test_images = np.fromfile("D:\DataSci_data\mnist\mnist_test_data",dtype=np.uint8)
    test_images = test_images.reshape(10000,2025)
    test_labels = np.fromfile("D:\DataSci_data\mnist\mnist_test_label",dtype=np.uint8)
    test_labels = test_labels.reshape(10000,1)
 
    time_2 = time.time()
    print ('Read Data cost ', time_2 - time_1, ' second')
    print (len(test_images[0]))
    print ('Training...')

    w10, w20, hid_offset0, out_offset0 = BPClassify(train_images, train_labels,test_times)

    time_3 = time.time()
    print ('Training cost ', time_3 - time_2, ' second')
    print ('Testing...')
    predict = BPtest(test_images,w10,w20,hid_offset0,out_offset0)
    time_4 = time.time()
    print ('Testing cost ', time_4 - time_3, ' second')
    score = accuracy_score(test_labels, predict)
    print ('The accrucy score is ',score)


if __name__ == '__main__':
    main()