import numpy as np
import tensorflow as tf

tf.reset_default_graph()
batch_size=3
learning_rate=0.01
z_dim = 2
noise_dim=3
gen_hidden_dim=20
data_dim=1
disc_hidden_dim=20
#z=np.random.randn(3)
#x=np.random.randn(3)
z=np.sqrt(2)*np.random.randn(5*batch_size, z_dim)
def problikelihood(z):
    return tf.random_gamma(shape=[data_dim], alpha=1, beta=3+tf.pow(tf.maximum(0.0,z[:,0]),3)+tf.pow(tf.maximum(0.0,z[:,1]),3))
#x=tf.random_gamma([batch_size,1],alpha=1,beta=3+np.power(tf.maximum(0.0,z[0]),3)+np.power(tf.maximum(0.0,z[1]),3),dtype=tf.float32)
#x=np.random.exponential(size=(batch_size, data_dim), scale=3+np.power(tf.maximum(0.0,z[:,0]),3)+np.power(tf.maximum(0.0,z[:,1]),3))
noise=tf.random_normal([noise_dim],mean=0.0,stddev=1.0,dtype=tf.float32)
swag=tf.random_normal([noise_dim],mean=0.0,stddev=1.0,dtype=tf.float32)
#noise=np.random.randn(batch_size, noise_dim)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(z)
    x=problikelihood(z)
    print(x)
