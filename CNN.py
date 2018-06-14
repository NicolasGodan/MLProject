import tensorflow as tf
import numpy as np

def calScore(v_x, v_y):
    global Pred
    y_predicted = sess.run(Pred, feed_dict={x: v_x, keep_prob: 1})
    Pred_True = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(v_y, 1))
    Score = tf.reduce_mean(tf.cast(Pred_True, tf.float32))
    Ans = sess.run(Score, feed_dict={x: v_x, y: v_y, keep_prob: 1})
    return Ans

def conv2d(x, W):
    # strides=[1,x_movement,y_movement,1]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def setWeight(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def setBias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def next_batch(a, b, N, n_batch):
    indices = np.random.choice(N, n_batch)
    return a[indices], b[indices]

# load mnist data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_num = 60000 #The number of figures
test_num = 10000
fig_w = 45       #width of each figure
train_images = np.fromfile("mnist_train_data",dtype=np.uint8).astype(np.float32)
train_label = np.fromfile("mnist_train_label",dtype=np.uint8)
test_images = np.fromfile("mnist_test_data",dtype=np.uint8).astype(np.float32)
test_label = np.fromfile("mnist_test_label",dtype=np.uint8)

#print(test_label[0:10])
train_label = np.eye(10)[train_label].astype(np.float32)
test_label = np.eye(10)[test_label].astype(np.float32)

train_images = train_images.reshape(train_num,fig_w*fig_w)
test_images = test_images.reshape(test_num,fig_w*fig_w)

#np.set_printoptions(threshold=np.inf)
#print(train_images[0])

x = tf.placeholder("float", shape=[None, 2025])
y = tf.placeholder("float", shape=[None, 10])
keep_prob = tf.placeholder("float")
# reshape data
x_image = tf.reshape(x, [-1, 45, 45, 1])

# convolutional layer 1
# transfer a 5*5*1 imagine into 32 sequence
Weight_conv1 = setWeight([5, 5, 1, 32])
Bias_conv1 = setBias([32])
# input a imagine and make a 5*5*1 to 32 with stride=1*1
h_conv1 = tf.nn.relu(conv2d(x_image, Weight_conv1) + Bias_conv1)  # output size 45*45*32
h_pool1 = max_pool_2x2(h_conv1)  # output size 23*23*32

# convolutional layer 2
# transfer a 5*5*32 imagine into 64 sequence
Weight_conv2 = setWeight([5, 5, 32, 64])
Bias_conv2 = setBias([64])
# input a imagine and make a 5*5*32 to 64 with stride=1*1
h_conv2 = tf.nn.relu(conv2d(h_pool1, Weight_conv2) + Bias_conv2)  # output size 23*23*64
h_pool2 = max_pool_2x2(h_conv2)  # output size 12*12*64

# function layer 1
Weight_fc1 = setWeight([12 * 12 * 64, 1024])
Bias_fc1 = setBias([1024])
# reshape the image from 7,7,64 into a flat (7*7*64)
h_pool2_flat = tf.reshape(h_pool2, [-1, 12 * 12 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, Weight_fc1) + Bias_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# function layer 2
Weight_fc2 = setWeight([1024, 10])
Bias_fc2 = setBias([10])
Pred = tf.nn.softmax(tf.matmul(h_fc1_drop, Weight_fc2) + Bias_fc2)

# loss calculation
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(Pred), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# init session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(5000):
    #print(i)
    batch0, batch1 = next_batch(train_images, train_label, train_num, 50)
    if i % 100 == 0:
        train_accuracy = calScore(test_images[0:1000], test_label[0:1000])
        print("step %d, training accuracy %.5f" % (i, train_accuracy))
    _, err, ppred = sess.run([train_step, cross_entropy, Pred], feed_dict={x: batch0, y: batch1, keep_prob: 0.5})
    #print('pred', ppred)

print("final accuracy %.5f" % calScore(test_images[0:1000], test_label[0:1000]))