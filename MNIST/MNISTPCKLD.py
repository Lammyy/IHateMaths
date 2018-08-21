import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from matplotlib import pylab as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import statsmodels.api as sm
import time
tf.reset_default_graph()
from tensorflow.examples.tutorials.mnist import input_data
#from official.mnist import dataset

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 32
z_dim = 24
noise_dim = 4
data_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
#number of nodes in hidden layer
h_dim = 128
lr = 1e-3
epsilon=1E-18
ratio_hidden_dim1=200
ratio_hidden_dim2=400
gen_hidden_dim1=200
batch_size=32
gen_hidden_dim2=400
like_hidden_dim1=500
like_hidden_dim2=1000
learning_rate_p=0.0005
learning_rate_d=0.00001
def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig
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
    'like_hidden1': tf.Variable(xavier_init(z_dim, like_hidden_dim1)),
    'like_hidden2': tf.Variable(xavier_init(like_hidden_dim1,like_hidden_dim2)),
    'like_out': tf.Variable(xavier_init(like_hidden_dim2, data_dim)),
    #ratio_hidden11 and ratio_hidden12 work on z input
    'ratio_hidden11': tf.Variable(xavier_init(z_dim, ratio_hidden_dim1)),
    'ratio_hidden12': tf.Variable(xavier_init(ratio_hidden_dim1, ratio_hidden_dim2)),
    #ratio_hidden21 and ratio_hidden22 work on x input
    'ratio_hidden21': tf.Variable(xavier_init(data_dim, ratio_hidden_dim1)),
    'ratio_hidden22': tf.Variable(xavier_init(ratio_hidden_dim1, ratio_hidden_dim2)),
    #ratio_hidden31, ratio_hidden32 and ratio_out work on concatenated z and x
    'ratio_hidden31': tf.Variable(xavier_init(ratio_hidden_dim2+ratio_hidden_dim2, ratio_hidden_dim1)),
    'ratio_hidden32': tf.Variable(xavier_init(ratio_hidden_dim1, ratio_hidden_dim2)),
    'ratio_out': tf.Variable(xavier_init(ratio_hidden_dim2, 1))
}
biases = {
    'post_hidden11': tf.Variable(tf.zeros([gen_hidden_dim1])),
    'post_hidden12': tf.Variable(tf.zeros([gen_hidden_dim2])),
    'post_hidden2': tf.Variable(tf.zeros([gen_hidden_dim2])),
    'post_hidden31': tf.Variable(tf.zeros([gen_hidden_dim1])),
    'post_hidden32': tf.Variable(tf.zeros([gen_hidden_dim2])),
    'post_out': tf.Variable(tf.zeros([z_dim])),
    'like_hidden1': tf.Variable(tf.zeros([like_hidden_dim1])),
    'like_hidden2': tf.Variable(tf.zeros([like_hidden_dim2])),
    'like_out': tf.Variable(tf.zeros([data_dim])),
    'ratio_hidden11': tf.Variable(tf.zeros([ratio_hidden_dim1])),
    'ratio_hidden12': tf.Variable(tf.zeros([ratio_hidden_dim2])),
    'ratio_hidden21': tf.Variable(tf.zeros([ratio_hidden_dim1])),
    'ratio_hidden22': tf.Variable(tf.zeros([ratio_hidden_dim2])),
    'ratio_hidden31': tf.Variable(tf.zeros([ratio_hidden_dim1])),
    'ratio_hidden32': tf.Variable(tf.zeros([ratio_hidden_dim2])),
    'ratio_out': tf.Variable(tf.zeros([1]))
}

def posterior(x, noise):
    hidden_layer11 = tf.nn.relu(tf.matmul(x, weights['post_hidden11'])+biases['post_hidden11'])
    hidden_layer12 = tf.nn.relu(tf.matmul(hidden_layer11, weights['post_hidden12'])+biases['post_hidden12'])
    hidden_layer2 = tf.nn.relu(tf.matmul(noise, weights['post_hidden2'])+biases['post_hidden2'])
    hidden_layer = tf.concat([hidden_layer12, hidden_layer2],axis=1)
    hidden_layer31 = tf.nn.relu(tf.matmul(hidden_layer, weights['post_hidden31'])+biases['post_hidden31'])
    hidden_layer32 = tf.nn.relu(tf.matmul(hidden_layer31, weights['post_hidden32'])+biases['post_hidden32'])
    out_layer = tf.matmul(hidden_layer32, weights['post_out'])+biases['post_out']
    return out_layer

def likelihood(z):
    hidden_layer1 = tf.nn.relu(tf.matmul(z, weights['like_hidden1'])+biases['like_hidden1'])
    hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer1, weights['like_hidden2'])+biases['like_hidden2'])
    hidden_layer3 = tf.nn.relu(tf.matmul(hidden_layer2, weights['like_hidden3'])+biases['like_hidden3'])
    logits = tf.matmul(hidden_layer3, weights['like_out'])+biases['like_out']
    out_layer = tf.nn.sigmoid(logits)
    return out_layer, logits

def ratiomator(z, x):
    hidden_layer11 = tf.nn.relu(tf.matmul(z, weights['ratio_hidden11'])+biases['ratio_hidden11'])
    hidden_layer12 = tf.nn.relu(tf.matmul(hidden_layer11, weights['ratio_hidden12'])+biases['ratio_hidden12'])
    hidden_layer21 = tf.nn.relu(tf.matmul(x, weights['ratio_hidden21'])+biases['ratio_hidden21'])
    hidden_layer22 = tf.nn.relu(tf.matmul(hidden_layer21, weights['ratio_hidden22'])+biases['ratio_hidden22'])
    hidden_layer = tf.concat([hidden_layer12, hidden_layer22], axis=1)
    hidden_layer31 = tf.nn.relu(tf.matmul(hidden_layer, weights['ratio_hidden31'])+biases['ratio_hidden31'])
    hidden_layer32 = tf.nn.relu(tf.matmul(hidden_layer31, weights['ratio_hidden32'])+biases['ratio_hidden32'])
    out_layer = tf.matmul(hidden_layer32, weights['ratio_out'])+biases['ratio_out']
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer

with tf.device('/gpu:0'):
    #q*(x)
    x_input = tf.placeholder(tf.float32, shape=[None, data_dim], name='x_input')
    #pi(eps)
    noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='noise_input')
    #p(z)
    prior_input = tf.placeholder(tf.float32, shape=[None, z_dim], name='ratio_input')
    z_samp = tf.placeholder(tf.float32, shape=[None, z_dim], name = 'z_samp')
    #G(eps;x)
    post_sample = posterior(x_input, noise_input)
    X_samples, _ = likelihood(z_samp)
    ratio_prior = ratiomator(prior_input, x_input)
    ratio_post = ratiomator(post_sample, x_input)
    _, data_logits = likelihood(post_sample)
    likeli = tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=data_logits, labels=x_input),
        axis=1
    )
    ratio_loss = -tf.reduce_mean(tf.log((ratio_post+epsilon)/(1-ratio_post+epsilon)))+tf.reduce_mean((ratio_prior)/(1-ratio_prior))

    nelbo=tf.reduce_mean(likeli)+tf.reduce_mean(tf.log(tf.divide(ratio_post+epsilon,1-ratio_post+epsilon)))

    post_vars = [weights['post_hidden11'],weights['post_hidden12'], weights['post_hidden2'], weights['post_hidden31'], weights['post_hidden32'], weights['post_out'],
    biases['post_hidden11'], biases['post_hidden12'], biases['post_hidden2'], biases['post_hidden31'], biases['post_hidden32'], biases['post_out']]

    ratio_vars = [weights['ratio_hidden11'], weights['ratio_hidden12'], weights['ratio_hidden21'], weights['ratio_hidden22'], weights['ratio_hidden31'], weights['ratio_hidden32'], weights['ratio_out'],
    biases['ratio_hidden11'], biases['ratio_hidden12'], biases['ratio_hidden21'], biases['ratio_hidden22'], biases['ratio_hidden31'], biases['ratio_hidden32'], biases['ratio_out']]
    like_vars = [weights['like_hidden1'], weights['like_hidden2'], weights['like_out'], biases['like_hidden1'], biases['like_hidden2'], biases['like_out']]
    train_elbo = tf.train.AdamOptimizer(learning_rate=learning_rate_p).minimize(nelbo, var_list=post_vars+like_vars)
    train_ratio = tf.train.AdamOptimizer(learning_rate=learning_rate_d).minimize(ratio_loss, var_list=ratio_vars)

#if no NVIDIA CUDA take out config=...
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
#with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #Pre-Train ratioriminator
    for i in range(5001):
        z=np.random.randn(batch_size, z_dim)
        xin, _ = mnist.train.next_batch(mb_size)
        noise=np.random.randn(batch_size, noise_dim)
        feed_dict = {prior_input: z, x_input: xin, noise_input: noise}
        _, rl = sess.run([train_ratio, ratio_loss], feed_dict=feed_dict)
        if i % 500 == 0:
            print('Step %i: ratioriminator Loss: %f' % (i, rl))
    for j in range(50001):
        print('Iteration %i' % (j))
        #Train ratioriminator
        for i in range(81):
            #Prior sample N(0,I_2x2)
            z=np.random.randn(batch_size, z_dim)
            xin, _ = mnist.train.next_batch(mb_size)
            noise=np.random.randn(batch_size, noise_dim)
            feed_dict = {prior_input: z, x_input: xin, noise_input: noise}
            _, rl = sess.run([train_ratio, ratio_loss], feed_dict=feed_dict)
            if i % 80 == 0:
                print('Step %i: ratioriminator Loss: %f' % (i, rl))
        #Train Posterior on the 5 values of x specified at the start
        for k in range(1):
            xin, _ = mnist.train.next_batch(mb_size)
            noise=np.random.randn(batch_size, noise_dim)
            feed_dict = {x_input: xin, noise_input: noise}
            _, nelboo = sess.run([train_elbo, nelbo], feed_dict=feed_dict)
            #if k % 1000 == 0 or k ==1:
            print('Step %i: NELBO: %f' % (k, nelboo))
        if j % 100 == 0:
            samples = sess.run(X_samples, feed_dict={z_samp: np.random.randn(16, z_dim)})
            fig = plot(samples)
            plt.savefig('FiguresMNISTPCKLD\Fig %i'%(j), bbox_inches='tight')
            plt.close(fig)
