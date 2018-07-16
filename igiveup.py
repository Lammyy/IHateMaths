import os

# Comment

from matplotlib import pylab as plt
import theano
from theano import tensor as T
import numpy as np
#create a regular grid in weight space for visualisation
xmin = -5
xmax = 5
xrange = np.linspace(xmin,xmax,300)
x = np.repeat(xrange[:,None],300,axis=1)
x = np.concatenate([[x.flatten()],[x.T.flatten()]])

prior_variance = 2
logprior = -(x**2).sum(axis=0)/2/prior_variance
plt.contourf(xrange, xrange, logprior.reshape(300,300), cmap='gray');
plt.axis('square');
plt.xlim([xmin,xmax])
plt.ylim([xmin,xmax]);
#let the likelihood be a simple exponential distribution where the
#mean is parametrised by the two hidden variables x_1 and x_2

def likelihood(x, y, beta_0=3, beta_1=1):
    beta = beta_0 + (beta_1*(x**3).clip(0,np.Inf).sum(axis=0))
    return -np.log(beta) - y/beta

y = [None]*5
y[0] = 0
y[1] = 5
y[2] = 8
y[3] = 12
y[4] = 50

llh = [None]*5
llh[0] = likelihood(x, y[0])
llh[1] = likelihood(x, y[1])
llh[2] = likelihood(x, y[2])
llh[3] = likelihood(x, y[3])
llh[4] = likelihood(x, y[4])
#plotting posteriors for different observed y values
plt.subplots(figsize=(20,4))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.contourf(xrange, xrange, np.exp(logprior + llh[i]).reshape(300,300), cmap='gray')
    plt.title('y={}'.format(y[i]))
    plt.axis('square');
    plt.xlim([xmin,xmax])
    plt.ylim([xmin,xmax])

    from lasagne.utils import floatX

    from lasagne.layers import (
        InputLayer,
        DenseLayer,
        NonlinearityLayer,
        ElemwiseSumLayer,
        ReshapeLayer,
    )
    from lasagne.nonlinearities import sigmoid

    #defines a 'generator' network with two inputs, noise z and observation y

    def build_G(input_z_var=None, input_y_var=None, num_z = 3):

        input_y_layer = InputLayer(input_var=input_y_var, shape=(None, 1))

        input_z_layer = InputLayer(input_var=input_z_var, shape=(None, num_z))

        network1 = DenseLayer(incoming = input_y_layer, num_units=10)

        network1 = DenseLayer(incoming = network1, num_units=20)

        network2 =  DenseLayer(incoming = input_z_layer, num_units=20)

        network = ElemwiseSumLayer(incomings=(network1,network2))

        network = DenseLayer(incoming=network, num_units=10)

        network = DenseLayer(incoming=network, num_units=20)

        network = DenseLayer(incoming = network, num_units=2, nonlinearity=None)

        return network

    #defines the 'discriminator network'
    def build_D(input_x_var=None, input_y_var=None):

        input_x_layer = InputLayer(input_var=input_x_var, shape = (None, 2))

        input_y_layer = InputLayer(input_var=input_y_var, shape = (None, 1))

        network1 = DenseLayer(incoming = input_x_layer, num_units=10)

        network1 = DenseLayer(incoming = network1, num_units=20)

        network2 = DenseLayer(incoming = input_y_layer, num_units=10)

        network2 = DenseLayer(incoming = network2, num_units=20)

        network = ElemwiseSumLayer(incomings=(network1, network2))

        network = DenseLayer(incoming = network, num_units=10)

        network = DenseLayer(incoming = network, num_units=20)

        network = DenseLayer(incoming = network, num_units=1, nonlinearity=None)

        normalised = NonlinearityLayer(incoming = network, nonlinearity = sigmoid)

        return { 'unnorm':network, 'norm':normalised }
        from lasagne.layers import get_output, get_all_params
        from theano.printing import debugprint
        from lasagne.updates import adam
        from theano.tensor.shared_randomstreams import RandomStreams

        #variables for latent, observed and GAN noise variables
        x_var = T.matrix('hidden variable')
        y_var = T.vector('observation')
        z_var = T.matrix('GAN noise')

        #theano variables for things like batchsize, learning rate, etc.
        prior_variance_var = T.scalar('prior variance')
        learningrate_var = T.scalar('learning rate')

        #random numbers for sampling from the prior or from the GAN
        srng = RandomStreams(seed=1337)
        z_rnd = srng.normal((y_var.shape[0],3))
        prior_rnd = srng.normal((y_var.shape[0],2))

        #instantiating the G and D networks
        generator = build_G(input_z_var=z_var, input_y_var=y_var.dimshuffle(0,'x'))

        #these expressions are random samples from the generator and the prior, respectively
        samples_from_generator = theano.clone(get_output(generator), replace={z_var: z_rnd})
        samples_from_prior = prior_rnd*T.sqrt(prior_variance_var)

        evaluate_generator = theano.function(
            [z_var, y_var],
            get_output(generator),
            allow_input_downcast=True
        )

        sample_generator = theano.function(
            [y_var],
            samples_from_generator,
            allow_input_downcast=True,
        )

        sample_prior = theano.function(
            [y_var, prior_variance_var],
            samples_from_prior,
            allow_input_downcast=True,
        )

        #showing q when initialised, without any training

y_test = np.array(y)
N_samples = 100
samples = sample_generator(np.repeat(y_test,100))
samples = samples.reshape(y_test.shape[0], N_samples, 2)

#plotting posteriors for different observed y values
plt.subplots(figsize=(20,4))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.contourf(xrange, xrange, np.exp(logprior + llh[i]).reshape(300,300), cmap='gray')
    plt.title('y={}'.format(y[i]))
    plt.axis('square');
    plt.plot(samples[i,:,0],samples[i,:,1],'r.')
    plt.xlim([xmin,xmax])
    plt.ylim([xmin,xmax])


    #compiling theano functions for D#compilin

    discriminator = build_D(input_x_var=x_var, input_y_var=y_var.dimshuffle(0,'x'))

    #discriminator output for synthetic samples, both normalised and unnormalised (after/before sigmoid)
    D_of_G = theano.clone(get_output(discriminator['norm']), replace={x_var:samples_from_generator})
    s_of_G = theano.clone(get_output(discriminator['unnorm']), replace={x_var:samples_from_generator})

    #discriminator output for real samples from the prior
    D_of_prior = theano.clone(get_output(discriminator['norm']), replace={x_var:samples_from_prior})

    #loss of discriminator - simple binary cross-entropy loss
    loss_D = -T.log(D_of_G).mean() - T.log(1-D_of_prior).mean()

    evaluate_discriminator = theano.function(
        [x_var, y_var],
        get_output([discriminator['unnorm'],discriminator['norm']]),
        allow_input_downcast = True
    )
    #log likelihood for each synthetic w sampled from the generator
    beta_0 = 3
    beta_1 = 1
    beta_expr = (beta_0 + beta_1*(T.nnet.relu(samples_from_generator)**3).sum(axis=1))
    log_likelihood = (- T.log(beta_expr) - y_var/beta_expr)

    #loss for G is the sum of unnormalised discriminator output and the negative log likelihood
    loss_G = s_of_G.mean() - log_likelihood.mean()
    #this is to evaluate the log-likelihood of an arbitrary set of x
    beta_expr_for_x = (beta_0 + beta_1*(T.nnet.relu(x_var)**3).sum(axis=1)).dimshuffle(0,'x')
    log_likelihood_for_x = - T.log(beta_expr_for_x) - y_var.dimshuffle('x',0)/beta_expr_for_x

    evaluate_loglikelihood = theano.function(
            [x_var, y_var],
            log_likelihood_for_x,
            allow_input_downcast = True
        )
    #testing if numpy and theano give the same likelihoods

    llh_theano = evaluate_loglikelihood(x.T, y_test)
    plt.subplots(figsize=(20,8))
    for i in range(5):
        plt.subplot(2,5,i+1)
        plt.contourf(xrange, xrange, llh[i].reshape(300,300), cmap='gray')
        plt.title('y={}'.format(y[i]))
        plt.axis('square');
        plt.xlim([xmin,xmax])
        plt.ylim([xmin,xmax])

        plt.subplot(2,5,5+i+1)
        plt.contourf(xrange, xrange, llh_theano[:,i].reshape(300,300), cmap='gray')
        plt.title('y={}'.format(y[i]))
        plt.axis('square');
        plt.xlim([xmin,xmax])
        plt.ylim([xmin,xmax])

        assert np.allclose(llh_theano[:,i], llh[i])
        #compiling theano functions for training D:

        params_D = get_all_params(discriminator['norm'], trainable=True)

        updates_D = adam(
            loss_D,
            params_D,
            learning_rate = learningrate_var
        )

        train_D = theano.function(
            [y_var, prior_variance_var, learningrate_var],
            loss_D,
            updates = updates_D,
            allow_input_downcast = True
        )
        #pretrain the discriminator for randomly initialised q to get training starting quicker

        learning_rate = 0.001
        for i in range(100):
            train_D(np.repeat(y_test,100), prior_variance, learning_rate)
            print(train_D(np.repeat(y_test,100), prior_variance, 0))
            N_samples = 100
            q_samples = sample_generator(np.repeat(y_test,100))
            q_samples = q_samples.reshape(y_test.shape[0], N_samples, 2)

            prior_samples = sample_prior(np.repeat(y_test,100), prior_variance)
            prior_samples = prior_samples.reshape(y_test.shape[0], N_samples, 2)

            plt.subplots(figsize=(20,4))
            for i in range(5):
                foobar = evaluate_discriminator(x.T, y[i]*np.ones(90000))[0]
                plt.subplot(1,5,i+1)
                plt.contourf(xrange, xrange, foobar[:,0].reshape(300,300).T, cmap='gray')
                plt.title('y={}'.format(y[i]))
                plt.axis('square');
                plt.plot(q_samples[i,:,0],q_samples[i,:,1],'r.')
                plt.plot(prior_samples[i,:,0],prior_samples[i,:,1],'b.')

                plt.xlim([xmin,xmax])
                plt.ylim([xmin,xmax])
                #compile theano functions for training q/G

params_G = get_all_params(generator, trainable=True)
updates_G = adam(
                  loss_G,
                  params_G,
                  learning_rate = learningrate_var
              )

train_G = theano.function(
                  [y_var, learningrate_var],
                  loss_G,
                  updates = updates_G,
                  allow_input_downcast = True
              )
              #main training loop - will need more iterations, or run this cell several times

learning_rate = 0.0003
for i in range(300):
                  train_D(np.repeat(y_test,500), prior_variance, learning_rate)
                  print(train_G(np.repeat(y_test,500), learning_rate))
