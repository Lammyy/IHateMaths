import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from matplotlib import pylab as plt
import seaborn as sns
import statsmodels.api as sm
import time
tf.reset_default_graph()
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 32
z_dim = 3
noise_dim = 4
data_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
#number of nodes in hidden layer
h_dim = 128
lr = 1e-3
epsilon=1E-18
disc_hidden_dim1=40
disc_hidden_dim2=80
gen_hidden_dim1=40
gen_hidden_dim2=80


def xavier_init(fan_in, fan_out, constant=1):
    low=-constant*np.sqrt(6.0/(fan_in+fan_out))
    high=constant*np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in, fan_out),minval=low,maxval=high, dtype=tf.float32)

def conv2d(x, W):
    #padding = SAME means output feature map has same spatial dimensions
    #as input feature map (yes PADDING) as opposed to VALID which only outputs
    #outputs dependent on input (no PADDING)
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
#simpler maxpool2d function with 2x2 window moving 2 pixels at a time
#max
def maxpool2d(x):
    #                        size of window   movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
weights = {
    #post_hidden11 and post_hidden12 work on the x input
    'post_hidden11': tf.Variable(xavier_init(data_dim, gen_hidden_dim1)),
    'post_hidden12': tf.Variable(xavier_init(gen_hidden_dim1, gen_hidden_dim2)),
    #post_hidden2 works on noise input
    'post_hidden2': tf.Variable(xavier_init(noise_dim, gen_hidden_dim2)),
    #post_hidden31, post_hidden32 and post_out work on concatenated x and noise
    'post_hidden31': tf.Variable(xavier_init(gen_hidden_dim2+gen_hidden_dim2, gen_hidden_dim1)),
    'post_hidden32': tf.Variable(xavier_init(gen_hidden_dim1, gen_hidden_dim2)),
    'post_out': tf.Variable(xavier_init(gen_hidden_dim2, z_dim)),
    #disc_hidden11 and disc_hidden12 work on z input
    'disc_hidden11': tf.Variable(xavier_init(z_dim, disc_hidden_dim1)),
    'disc_hidden12': tf.Variable(xavier_init(disc_hidden_dim1, disc_hidden_dim2)),
    #disc_hidden21 and disc_hidden22 work on x input
    'disc_hidden21': tf.Variable(xavier_init(data_dim, disc_hidden_dim1)),
    'disc_hidden22': tf.Variable(xavier_init(disc_hidden_dim1, disc_hidden_dim2)),
    #disc_hidden31, disc_hidden32 and disc_out work on concatenated z and x
    'disc_hidden31': tf.Variable(xavier_init(disc_hidden_dim2+disc_hidden_dim2, disc_hidden_dim1)),
    'disc_hidden32': tf.Variable(xavier_init(disc_hidden_dim1, disc_hidden_dim2)),
    'disc_out': tf.Variable(xavier_init(disc_hidden_dim2, 1))
}
biases = {
    'post_hidden11': tf.Variable(tf.zeros([gen_hidden_dim1])),
    'post_hidden12': tf.Variable(tf.zeros([gen_hidden_dim2])),
    'post_hidden2': tf.Variable(tf.zeros([gen_hidden_dim2])),
    'post_hidden31': tf.Variable(tf.zeros([gen_hidden_dim1])),
    'post_hidden32': tf.Variable(tf.zeros([gen_hidden_dim2])),
    'post_out': tf.Variable(tf.zeros([z_dim])),
    'disc_hidden11': tf.Variable(tf.zeros([disc_hidden_dim1])),
    'disc_hidden12': tf.Variable(tf.zeros([disc_hidden_dim2])),
    'disc_hidden21': tf.Variable(tf.zeros([disc_hidden_dim1])),
    'disc_hidden22': tf.Variable(tf.zeros([disc_hidden_dim2])),
    'disc_hidden31': tf.Variable(tf.zeros([disc_hidden_dim1])),
    'disc_hidden32': tf.Variable(tf.zeros([disc_hidden_dim2])),
    'disc_out': tf.Variable(tf.zeros([1]))
}

def posterior(x, noise):
    x=tf.flatten(x)
    hidden_layer11 = tf.nn.relu(tf.matmul(x, weights['post_hidden11'])+biases['post_hidden11'])
    hidden_layer12 = tf.nn.relu(tf.matmul(hidden_layer11, weights['post_hidden12'])+biases['post_hidden12'])
    hidden_layer2 = tf.nn.relu(tf.matmul(noise, weights['post_hidden2'])+biases['post_hidden2'])
    hidden_layer = tf.concat([hidden_layer12, hidden_layer2],axis=1)
    hidden_layer31 = tf.nn.relu(tf.matmul(hidden_layer, weights['post_hidden31'])+biases['post_hidden31'])
    hidden_layer32 = tf.nn.relu(tf.matmul(hidden_layer31, weights['post_hidden32'])+biases['post_hidden32'])
    out_layer = tf.matmul(hidden_layer32, weights['post_out'])+biases['post_out']
    return out_layer

def discriminator(z, x):
    x=tf.flatten(x)
    hidden_layer11 = tf.nn.relu(tf.matmul(z, weights['disc_hidden11'])+biases['disc_hidden11'])
    hidden_layer12 = tf.nn.relu(tf.matmul(hidden_layer11, weights['disc_hidden12'])+biases['disc_hidden12'])
    hidden_layer21 = tf.nn.relu(tf.matmul(x, weights['disc_hidden21'])+biases['disc_hidden21'])
    hidden_layer22 = tf.nn.relu(tf.matmul(hidden_layer21, weights['disc_hidden22'])+biases['disc_hidden22'])
    hidden_layer = tf.concat([hidden_layer12, hidden_layer22], axis=1)
    hidden_layer31 = tf.nn.relu(tf.matmul(hidden_layer, weights['disc_hidden31'])+biases['disc_hidden31'])
    hidden_layer32 = tf.nn.relu(tf.matmul(hidden_layer31, weights['disc_hidden32'])+biases['disc_hidden32'])
    out_layer = tf.matmul(hidden_layer32, weights['disc_out'])+biases['disc_out']
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer

with tf.device('/gpu:0'):
    #q*(x)
    x_input = tf.placeholder(tf.float32, shape=[None, data_dim, data_dim], name='x_input')
    #pi(eps)
    noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='noise_input')
    #p(z)
    prior_input = tf.placeholder(tf.float32, shape=[None, z_dim], name='disc_input')
    #G(eps;x)
    post_sample = posterior(x_input, noise_input)
    #p(x|z)
    xlike = problikelihood(prior_input)
    disc_prior = discriminator(prior_input, xlike)
    disc_post = discriminator(post_sample, x_input)

    disc_loss = -tf.reduce_mean(tf.log(disc_post+epsilon))-tf.reduce_mean(tf.log(1.0-disc_prior+epsilon))

    nelbo=tf.reduce_mean(tf.log(tf.divide(disc_post+epsilon,1-disc_post+epsilon)))

    post_vars = [weights['post_hidden11'],weights['post_hidden12'], weights['post_hidden2'], weights['post_hidden31'], weights['post_hidden32'], weights['post_out'],
    biases['post_hidden11'], biases['post_hidden12'], biases['post_hidden2'], biases['post_hidden31'], biases['post_hidden32'], biases['post_out']]

    disc_vars = [weights['disc_hidden11'], weights['disc_hidden12'], weights['disc_hidden21'], weights['disc_hidden22'], weights['disc_hidden31'], weights['disc_hidden32'], weights['disc_out'],
    biases['disc_hidden11'], biases['disc_hidden12'], biases['disc_hidden21'], biases['disc_hidden22'], biases['disc_hidden31'], biases['disc_hidden32'], biases['disc_out']]

    train_elbo = tf.train.AdamOptimizer(learning_rate=learning_rate_p).minimize(nelbo, var_list=post_vars)
    train_disc = tf.train.AdamOptimizer(learning_rate=learning_rate_d).minimize(disc_loss, var_list=disc_vars)

#if no NVIDIA CUDA take out config=...
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
#with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #Pre-Train Discriminator
    for i in range(5001):
        z=np.random.randn(batch_size, z_dim)
        xin, _ = mnist.train.next_batch(mb_size)
        noise=np.random.randn(5*batch_size, noise_dim)
        feed_dict = {prior_input: z, x_input: xin, noise_input: noise}
        _, dl = sess.run([train_disc, disc_loss], feed_dict=feed_dict)
        if i % 500 == 0:
            print('Step %i: Discriminator Loss: %f' % (i, dl))
