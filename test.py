import numpy as np
import tensorflow as tf
from scipy import stats
import matplotlib.pyplot as plt
tf.reset_default_graph()
batch_size=300
learning_rate=0.01
z_dim = 2
noise_dim=3
gen_hidden_dim=20
data_dim=1
disc_hidden_dim=20
#z=np.random.randn(3)
#x=np.random.randn(3)
z=np.sqrt(2)*np.random.randn(z_dim, 5*batch_size)
x=np.random.randn(z_dim, 5*batch_size)
zkernel=stats.gaussian_kde(z)
xkernel=stats.gaussian_kde(x)
samp=np.sqrt(2)*np.random.randn(z_dim, 5*batch_size)
print(np.mean(zkernel.logpdf(samp)-xkernel.logpdf(samp)))
