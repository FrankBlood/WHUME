# -*-coding:utf8-*-
import numpy as np
import cv2
import tensorflow as tf
import random

def label_one_hot(label_all):
    label_batch = []
    for label in label_all:
        one_hot = [0, 0]
        one_hot[label] = 1
        label_batch.append(one_hot)
    label_batch = np.array(label_batch)
    return label_batch

def load_data_from_file(file_path, batch_size, resize=(64, 64)): 
    X, Y= [], []
    fp = open(file_path, 'r')
    lines = []
    for line in fp.readlines():
        lines.append(line)
    random.shuffle(lines)

    for line in lines:
        img_path = line.strip().split()[0]
        x = cv2.imread(img_path)
        x = cv2.resize(x, resize)
        x = np.multiply(x, 1/255.0)
        y = np.array([int(line.strip().split()[1])])
        X.append(x)
        Y.append(y)
    fp.close()
    num = len(X)
    X = np.array(X)
    Y = np.array(Y)
    Y = label_one_hot(Y)
    X = X.astype("float32")
    Y = Y.astype("float64")
    for i in xrange(num/batch_size):
        yield np.array(X[i*batch_size:(i+1)*batch_size, :, :, :]), np.array(Y[i*batch_size:(i+1)*batch_size, :])
    
def deepnet():
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, [None, 64 ,64, 3])
    y_ = tf.placeholder("float", [None, 2])
    print '申请两个占位符'

    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    print '第一层卷积'

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    print h_pool2
    print '第二层卷积'

    W_fc1 = weight_variable([16*16*64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 16*16*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    print 'flatten'

    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    print 'dropout'

    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    print 'softmax'

    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    print 'cross_entropy'
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    print 'train_step'
    print tf.shape(y_conv)
    print tf.shape(y_)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    print 'correct_prediction'
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print 'accuracy'
    sess.run(tf.initialize_all_variables())
    print 'run'
    
    i = 0
    batch_size = 8
    print '训练模型'
    for (X_train, Y_train) in load_data_from_file('phone_train_set.txt', batch_size):
        #print np.shape(X_train), np.shape(Y_train), type(X_train[0,0,0,0]), type(Y_train[0,0])
        if i%10 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:X_train, y_:Y_train, keep_prob: 1.0})
            print (i, train_accuracy)
        train_step.run(feed_dict={x: X_train, y_: Y_train, keep_prob: 0.5})
        i = i + 1

    for (X_train, Y_train) in load_data_from_file('new_train.txt', batch_size):
        #print np.shape(X_train), np.shape(Y_train), type(X_train[0,0,0,0]), type(Y_train[0,0])
        if i%10 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:X_train, y_:Y_train, keep_prob: 1.0})
            print (i, train_accuracy)
        train_step.run(feed_dict={x: X_train, y_: Y_train, keep_prob: 0.5})
        i = i + 1

    print '测试模型'
    for (X_test, Y_test) in load_data_from_file('phone_test_set.txt', 200):
        print accuracy.eval(feed_dict={x: X_test, y_: Y_test, keep_prob: 1.0})
    # saver = tf.train.Saver()
    # saver.save(sess, './')

if __name__ == '__main__':
    deepnet()