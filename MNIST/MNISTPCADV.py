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
RECON=[None]*501
ISTHISLOSS=[None]*501
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
disc_hidden_dim1=200
disc_hidden_dim2=400
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

def doubleplot(samples1, samples2, loss, recondiff):
    fig = plt.figure(figsize=(8,4))
    gs = gridspec.GridSpec(4, 10)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples1):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    for i, sample in enumerate(samples2):
        ax = plt.subplot(gs[i+20])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    plt.text(-250, 45,'Estimator Loss: %f' %(loss), fontsize=17)
    plt.text(-250, 60,'Mean Reconstruction Error: %f' %(recondiff), fontsize=17)
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
    'like_hidden3': tf.Variable(xavier_init(like_hidden_dim2,like_hidden_dim2)),
    'like_out': tf.Variable(xavier_init(like_hidden_dim2, data_dim)),
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
    'like_hidden1': tf.Variable(tf.zeros([like_hidden_dim1])),
    'like_hidden2': tf.Variable(tf.zeros([like_hidden_dim2])),
    'like_hidden3': tf.Variable(tf.zeros([like_hidden_dim2])),
    'like_out': tf.Variable(tf.zeros([data_dim])),
    'disc_hidden11': tf.Variable(tf.zeros([disc_hidden_dim1])),
    'disc_hidden12': tf.Variable(tf.zeros([disc_hidden_dim2])),
    'disc_hidden21': tf.Variable(tf.zeros([disc_hidden_dim1])),
    'disc_hidden22': tf.Variable(tf.zeros([disc_hidden_dim2])),
    'disc_hidden31': tf.Variable(tf.zeros([disc_hidden_dim1])),
    'disc_hidden32': tf.Variable(tf.zeros([disc_hidden_dim2])),
    'disc_out': tf.Variable(tf.zeros([1]))
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

def discriminator(z, x):
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
    x_input = tf.placeholder(tf.float32, shape=[None, data_dim], name='x_input')
    #pi(eps)
    noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='noise_input')
    #p(z)
    prior_input = tf.placeholder(tf.float32, shape=[None, z_dim], name='disc_input')
    z_samp = tf.placeholder(tf.float32, shape=[None, z_dim], name = 'z_samp')
    #G(eps;x)
    post_sample = posterior(x_input, noise_input)
    X_samples, _ = likelihood(z_samp)
    disc_prior = discriminator(prior_input, x_input)
    disc_post = discriminator(post_sample, x_input)
    _, data_logits = likelihood(post_sample)
    likeli = tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=data_logits, labels=x_input),
        axis=1
    )
    disc_loss = -tf.reduce_mean(tf.log(disc_post+epsilon))-tf.reduce_mean(tf.log(1.0-disc_prior+epsilon))

    nelbo=tf.reduce_mean(likeli)+tf.reduce_mean(tf.log(tf.divide(disc_post+epsilon,1-disc_post+epsilon)))

    post_vars = [weights['post_hidden11'],weights['post_hidden12'], weights['post_hidden2'], weights['post_hidden31'], weights['post_hidden32'], weights['post_out'],
    biases['post_hidden11'], biases['post_hidden12'], biases['post_hidden2'], biases['post_hidden31'], biases['post_hidden32'], biases['post_out']]

    disc_vars = [weights['disc_hidden11'], weights['disc_hidden12'], weights['disc_hidden21'], weights['disc_hidden22'], weights['disc_hidden31'], weights['disc_hidden32'], weights['disc_out'],
    biases['disc_hidden11'], biases['disc_hidden12'], biases['disc_hidden21'], biases['disc_hidden22'], biases['disc_hidden31'], biases['disc_hidden32'], biases['disc_out']]
    like_vars = [weights['like_hidden1'], weights['like_hidden2'], weights['like_out'], biases['like_hidden1'], biases['like_hidden2'], biases['like_out']]
    train_elbo = tf.train.AdamOptimizer(learning_rate=learning_rate_p).minimize(nelbo, var_list=post_vars+like_vars)
    train_disc = tf.train.AdamOptimizer(learning_rate=learning_rate_d).minimize(disc_loss, var_list=disc_vars)

#if no NVIDIA CUDA take out config=...
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
#with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #Pre-Train Discriminator
    for i in range(5001):
        z=np.random.randn(batch_size, z_dim)
        xin, _ = mnist.train.next_batch(mb_size)
        noise=np.random.randn(batch_size, noise_dim)
        feed_dict = {prior_input: z, x_input: xin, noise_input: noise}
        _, dl = sess.run([train_disc, disc_loss], feed_dict=feed_dict)
        if i % 500 == 0:
            print('Step %i: Discriminator Loss: %f' % (i, dl))
    for j in range(50001):
        print('Iteration %i' % (j))
        #Train Discriminator
        for i in range(81):
            #Prior sample N(0,I_2x2)
            z=np.random.randn(batch_size, z_dim)
            xin, _ = mnist.train.next_batch(mb_size)
            noise=np.random.randn(batch_size, noise_dim)
            feed_dict = {prior_input: z, x_input: xin, noise_input: noise}
            _, dl = sess.run([train_disc, disc_loss], feed_dict=feed_dict)
            if i % 80 == 0:
                print('Step %i: Discriminator Loss: %f' % (i, dl))
                ISTHISLOSS[int(j/100)]=dl
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
            plt.savefig('FiguresMNISTPCADV\Fig %i'%(j), bbox_inches='tight')
            plt.close(fig)
            noise=np.random.randn(20, noise_dim)
            xin, _ = mnist.train.next_batch(20)
            z=sess.run(post_sample, feed_dict={x_input: xin, noise_input: noise})
            xout=sess.run(X_samples, feed_dict={z_samp: z})
            diff=np.mean(np.absolute(xin-xout))
            RECON[int(j/100)]=diff
            fig2 = doubleplot(xin, xout, ISTHISLOSS[int(j/100)], diff)
            plt.savefig('FiguresMNISTPCADV\Fig %i'%(j+1), bbox_inches='tight')
            plt.close(fig2)
    plt.subplot(2,1,1)
    plt.plot(RECON)
    plt.xlabel('Iterations (x100)')
    plt.ylabel('Avg Recon Error')
    plt.subplot(2,1,2)
    plt.plot(ISTHISLOSS)
    plt.xlabel('Iterations (x100)')
    plt.ylabel('Estimator Loss')
    plt.subplots_adjust(top=0.95,bottom=0.15,right=0.95,hspace=0.4)
    plt.savefig('FiguresMNISTPCADV\Diag Plot', bbox_inches='tight')
    plt.close()
