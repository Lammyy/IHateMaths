import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from matplotlib import pylab as plt
import seaborn as sns
tf.reset_default_graph()
#params
epsilon=0.0000001
batch_size=64
learning_rate_p=0.0002
learning_rate_d=0.0002
z_dim = 2
noise_dim=3
gen_hidden_dim1=30
gen_hidden_dim2=60
data_dim=1
ratio_hidden_dim1=30
ratio_hidden_dim2=60
#Stuff for making true posterior graph (copied from Huszar)
xmin = -5
xmax = 5
xrange = np.linspace(xmin,xmax,300)
x = np.repeat(xrange[:,None],300,axis=1)
x = np.concatenate([[x.flatten()],[x.T.flatten()]])
prior_variance = 2
logprior = -(x**2).sum(axis=0)/2/prior_variance
def likelihoodd(x, y, beta_0=3, beta_1=1):
    beta = beta_0 + (beta_1*(x**3).clip(0,np.Inf).sum(axis=0))
    return -np.log(beta) - y/beta
y = [None]*5
y[0] = 0
y[1] = 5
y[2] = 8
y[3] = 12
y[4] = 50
llh = [None]*5
llh[0] = likelihoodd(x, y[0])
llh[1] = likelihoodd(x, y[1])
llh[2] = likelihoodd(x, y[2])
llh[3] = likelihoodd(x, y[3])
llh[4] = likelihoodd(x, y[4])
#The values of x which we input into posterior later
xgen=np.array(y, dtype=np.float32)
#Xavier Initiaizlier
def xavier_init(fan_in, fan_out, constant=1):
    low=-constant*np.sqrt(6.0/(fan_in+fan_out))
    high=constant*np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in, fan_out),minval=low,maxval=high, dtype=tf.float32)

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
    'ratio_hidden11': tf.Variable(tf.zeros([ratio_hidden_dim1])),
    'ratio_hidden12': tf.Variable(tf.zeros([ratio_hidden_dim2])),
    'ratio_hidden21': tf.Variable(tf.zeros([ratio_hidden_dim1])),
    'ratio_hidden22': tf.Variable(tf.zeros([ratio_hidden_dim2])),
    'ratio_hidden31': tf.Variable(tf.zeros([ratio_hidden_dim1])),
    'ratio_hidden32': tf.Variable(tf.zeros([ratio_hidden_dim2])),
    'ratio_out': tf.Variable(tf.zeros([1]))
}
#likeli = p(x|z)
#def likelihood(z, x, beta_0=3., beta_1=1.):
#    beta = beta_0 + tf.reduce_sum(beta_1*tf.maximum(0.0, z**3), 1)
#    return -tf.log(beta) - x/beta

def problikelihood(z):
    return tf.transpose(tf.random_gamma(shape=[data_dim], alpha=1, beta=1/(3+tf.pow(tf.maximum(0.0,z[:,0]),3)+tf.pow(tf.maximum(0.0,z[:,1]),3))))
#post = q(z|x,eps)
def posterior(x, noise):
    hidden_layer11 = tf.nn.relu(tf.matmul(x, weights['post_hidden11'])+biases['post_hidden11'])
    hidden_layer12 = tf.nn.relu(tf.matmul(hidden_layer11, weights['post_hidden12'])+biases['post_hidden12'])
    hidden_layer2 = tf.nn.relu(tf.matmul(noise, weights['post_hidden2'])+biases['post_hidden2'])
    hidden_layer = tf.concat([hidden_layer12, hidden_layer2],axis=1)
    hidden_layer31 = tf.nn.relu(tf.matmul(hidden_layer, weights['post_hidden31'])+biases['post_hidden31'])
    hidden_layer32 = tf.nn.relu(tf.matmul(hidden_layer31, weights['post_hidden32'])+biases['post_hidden32'])
    out_layer = tf.matmul(hidden_layer32, weights['post_out'])+biases['post_out']
    return out_layer

def ratiomator(z, x):
    hidden_layer11 = tf.nn.relu(tf.matmul(z, weights['ratio_hidden11'])+biases['ratio_hidden11'])
    hidden_layer12 = tf.nn.relu(tf.matmul(hidden_layer11, weights['ratio_hidden12'])+biases['ratio_hidden12'])
    hidden_layer21 = tf.nn.relu(tf.matmul(x, weights['ratio_hidden21'])+biases['ratio_hidden21'])
    hidden_layer22 = tf.nn.relu(tf.matmul(hidden_layer21, weights['ratio_hidden22'])+biases['ratio_hidden22'])
    hidden_layer = tf.concat([hidden_layer12, hidden_layer22], axis=1)
    hidden_layer31 = tf.nn.relu(tf.matmul(hidden_layer, weights['ratio_hidden31'])+biases['ratio_hidden31'])
    hidden_layer32 = tf.nn.relu(tf.matmul(hidden_layer31, weights['ratio_hidden32'])+biases['ratio_hidden32'])
    out_layer = tf.matmul(hidden_layer32, weights['ratio_out'])+biases['ratio_out']
    out_layer = tf.nn.relu(out_layer)
    return out_layer
#Build Networks
#if no NVIDIA CUDA remove this line and unindent following lines
with tf.device('/gpu:0'):
    x_input = tf.placeholder(tf.float32, shape=[None, data_dim], name='x_input')
    noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='noise_input')
    prior_input = tf.placeholder(tf.float32, shape=[None, z_dim], name='prior_input')
    post_sample = posterior(x_input, noise_input)
    xlike = problikelihood(prior_input)
    ratio_prior = ratiomator(prior_input, xlike)
    ratio_post = ratiomator(post_sample, x_input)

    ratio_loss = -tf.reduce_mean(tf.log(ratio_post+epsilon))+tf.reduce_mean(ratio_prior)

    nelbo=tf.reduce_mean(tf.log(ratio_post+epsilon))

    post_vars = [weights['post_hidden11'],weights['post_hidden12'], weights['post_hidden2'], weights['post_hidden31'], weights['post_hidden32'], weights['post_out'],
    biases['post_hidden11'], biases['post_hidden12'], biases['post_hidden2'], biases['post_hidden31'], biases['post_hidden32'], biases['post_out']]

    ratio_vars = [weights['ratio_hidden11'], weights['ratio_hidden12'], weights['ratio_hidden21'], weights['ratio_hidden22'], weights['ratio_hidden31'], weights['ratio_hidden32'], weights['ratio_out'],
    biases['ratio_hidden11'], biases['ratio_hidden12'], biases['ratio_hidden21'], biases['ratio_hidden22'], biases['ratio_hidden31'], biases['ratio_hidden32'], biases['ratio_out']]

    train_elbo = tf.train.AdamOptimizer(learning_rate=learning_rate_p).minimize(nelbo, var_list=post_vars)
    train_ratio = tf.train.AdamOptimizer(learning_rate=learning_rate_d).minimize(ratio_loss, var_list=ratio_vars)


#if no NVIDIA CUDA take out config=...
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
#with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #Pre-Train ratiomator
    for i in range(8000):
        z=np.sqrt(2)*np.random.randn(5*batch_size, z_dim)
        xin=np.repeat(xgen,batch_size)
        xin=xin.reshape(5*batch_size, 1)
        noise=np.random.randn(5*batch_size, noise_dim)
        feed_dict = {prior_input: z, x_input: xin, noise_input: noise}
        _, dl = sess.run([train_ratio, ratio_loss], feed_dict=feed_dict)
        if i % 1000 == 0 or i == 1:
            print('Step %i: ratiomator Loss: %f' % (i, dl))
    for j in range(400):
        print('Iteration %i' % (j))
        #Train ratiomator
        for i in range(1001):
            #Prior sample N(0,I_2x2)
            z=np.sqrt(2)*np.random.randn(5*batch_size, z_dim)
            xin=np.repeat(xgen,batch_size)
            xin=xin.reshape(5*batch_size, 1)
            noise=np.random.randn(5*batch_size, noise_dim)
            feed_dict = {prior_input: z, x_input: xin, noise_input: noise}
            _, dl = sess.run([train_ratio, ratio_loss], feed_dict=feed_dict)
            if i % 1000 == 0 or i == 1:
                print('Step %i: ratiomator Loss: %f' % (i, dl))
        #Train Posterior on the 5 values of x specified at the start
        for k in range(1):
            xin=np.repeat(xgen,batch_size)
            xin=xin.reshape(5*batch_size, 1)
            noise=np.random.randn(5*batch_size, noise_dim)
            feed_dict = {x_input: xin, noise_input: noise}
            _, nelboo = sess.run([train_elbo, nelbo], feed_dict=feed_dict)
            #if k % 1000 == 0 or k ==1:
            print('Step %i: NELBO: %f' % (k, nelboo))

        if j % 10 == 0 or j == 1:
            sns.set_style('whitegrid')
            sns.set_context('poster')

            plt.subplots(figsize=(20,8))
            #make 5000 noise and 1000 of each x sample
            N_samples=1000
            noise=np.random.randn(5*N_samples, noise_dim).astype('float32')
            x_gen=np.repeat(xgen,N_samples)
            x_gen=x_gen.reshape(5*N_samples,1)
            #plug into posterior
            z_samples=posterior(x_gen,noise)
            z_samples=tf.reshape(z_samples,[xgen.shape[0], N_samples, 2]).eval()
            #print(z_samples)
            #Plots
            for i in range(5):
                plt.subplot(2,5,i+1)
                sns.kdeplot(z_samples[i,:,0], z_samples[i,:,1], cmap='Greens')
                #plt.scatter(z_samples[i,:,0],z_samples[i,:,1])
                plt.axis('square');
                plt.title('q(z|x={})'.format(y[i]))
                plt.xlim([xmin,xmax])
                plt.ylim([xmin,xmax])
                plt.xticks([])
                plt.yticks([]);
                plt.subplot(2,5,5+i+1)
                plt.contour(xrange, xrange, np.exp(logprior+llh[i]).reshape(300,300).T, cmap='Greens')
                plt.axis('square');
                plt.title('p(z|x={})'.format(y[i]))
                plt.xlim([xmin,xmax])
                plt.ylim([xmin,xmax])
                plt.xticks([])
                plt.yticks([]);
            plt.savefig('FiguresJCKL\Fig %i'%(j))
            plt.close()


    sns.set_style('whitegrid')
    sns.set_context('poster')

    plt.subplots(figsize=(20,8))
    #make 5000 noise and 1000 of each x sample
    N_samples=2000
    noise=np.random.randn(5*N_samples, noise_dim).astype('float32')
    x_gen=np.repeat(xgen,2000)
    x_gen=x_gen.reshape(10000,1)
    #plug into posterior
    z_samples=posterior(x_gen,noise)
    z_samples=tf.reshape(z_samples,[xgen.shape[0], N_samples, 2]).eval()
    #print(z_samples)
    #Plots
    for i in range(5):
        plt.subplot(2,5,i+1)
        sns.kdeplot(z_samples[i,:,0], z_samples[i,:,1], cmap='Greens')
        #plt.scatter(z_samples[i,:,0],z_samples[i,:,1])
        plt.axis('square');
        plt.title('q(z|x={})'.format(y[i]))
        plt.xlim([xmin,xmax])
        plt.ylim([xmin,xmax])
        plt.xticks([])
        plt.yticks([]);
        plt.subplot(2,5,5+i+1)
        plt.contour(xrange, xrange, np.exp(logprior+llh[i]).reshape(300,300).T, cmap='Greens')
        plt.axis('square');
        plt.title('p(z|x={})'.format(y[i]))
        plt.xlim([xmin,xmax])
        plt.ylim([xmin,xmax])
        plt.xticks([])
        plt.yticks([]);
    plt.show()
