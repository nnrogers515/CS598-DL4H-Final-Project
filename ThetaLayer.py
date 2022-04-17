import lasagne
import theano
import theano.tensor as T
import numpy as np

class ThetaLayer(lasagne.layers.Layer):
    def __init__(self, incomings, **kwargs):
        self.logsigma = incomings[1]
        self.mu = incomings[0]
        self.klterm = theano.function([self.logsigma, self.mu], 0.5 * (1 + T.mul(self.logsigma, 2) - (self.mu ** 2) - (T.exp(self.logsigma)**2)))


    def get_output_for(self, input, **kwargs):
        out = theano.function([self.mu, self.logsigma, input], (1 / (input * (T.exp(self.logsigma) * (2 * np.pi)**(1/2)))) * T.exp(-((T.log(input) - T.exp(self.logsigma))**2)/(2 * ( T.exp(self.logsigma)**2))))
        return out