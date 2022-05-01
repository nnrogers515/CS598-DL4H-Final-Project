import lasagne
import theano
import theano.tensor as T
import numpy as np

class ThetaLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, **kwargs):
        super().__init__(incomings, name="Theta")
        self.logsigma = incomings[1]
        self.mu = incomings[0]
        self.klterm = 0
        self.theta = self.add_param(lasagne.init.Constant(0), (1, self.logsigma.num_units), name='theta')

    def get_output_for(self, input, **kwargs):
        logsigma_in = self.logsigma.get_output_for(input)
        mu_in = self.mu.get_output_for(input)
        self.klterm = theano.function([logsigma_in, mu_in], 0.5 * (1 + T.mul(logsigma_in, 2) - (mu_in ** 2) - (T.exp(logsigma_in)**2)))
        out = theano.function([mu_in, logsigma_in, input], (1 / (input * (T.exp(logsigma_in) * (2 * np.pi)**(1/2)))) * T.exp(-((T.log(input) - mu_in)**2)/(2 * ( T.exp(logsigma_in)**2))))
        self.theta = lasagne.layers.ElemwiseMergeLayer([mu_in, logsigma_in, input], out, cropping="autocrop")
        return out

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], input_shapes[1][0])