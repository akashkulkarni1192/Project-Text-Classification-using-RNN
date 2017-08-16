import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import util as myutil


class RNN_class:
    def __init__(self, M, V):
        self.M = M  # hidden layer size
        self.V = V  # vocabulary size

    def fit(self, X, Y, learning_rate=10e-1, mu=0.99, activation=T.tanh, epochs=500):
        M = self.M
        V = self.V
        K = len(set(Y))
        X, Y = shuffle(X, Y)
        N_entries_valid = 10
        Xvalid, Yvalid = X[-N_entries_valid:], Y[-N_entries_valid:]
        X, Y = X[:-N_entries_valid], Y[:-N_entries_valid]
        N = len(X)

        # initialize weights
        Wx = myutil.init_weight(V, M)
        Wh = myutil.init_weight(M, M)
        bh = np.zeros(M)
        h0 = np.zeros(M)
        Wo = myutil.init_weight(M, K)
        bo = np.zeros(K)

        #Create theano variables
        thX, thY, py_x, prediction = self.set(Wx, Wh, bh, h0, Wo, bo, activation)

        cost = -T.mean(T.log(py_x[thY]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value() * 0) for p in self.params]
        lr = T.scalar('learning_rate') # Learning rate would be decremented by factor of 0.99 at each epcch

        updates = [
                      (p, p + mu * dp - lr * g) for p, dp, g in zip(self.params, dparams, grads)
                  ] + [
                      (dp, mu * dp - lr * g) for dp, g in zip(dparams, grads)
                  ]

        self.train_op = theano.function(
            inputs=[thX, thY, lr],
            outputs=[cost, prediction],
            updates=updates,
            allow_input_downcast=True,
        )

        costs = []
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            n_correct = 0
            cost = 0
            # stochastic gradient descent
            for j in range(N):
                c, p = self.train_op(X[j], Y[j], learning_rate)
                cost += c
                if p == Y[j]:
                    n_correct += 1
            # Decrement the learning rate
            learning_rate *= 0.9999

            # calculate validation accuracy
            n_correct_valid = 0
            for j in range(N_entries_valid):
                p = self.predict_op(Xvalid[j])
                if p == Yvalid[j]:
                    n_correct_valid += 1
            print("i:{0} cost:{1} correction_rate:{2}".format(i, cost, (float(n_correct) / N)))
            print("Validation correction rate:{0}".format (float(n_correct_valid) / Nvalid))
            costs.append(cost)
        plt.plot(costs)
        plt.show()


    def set(self, Wx, Wh, bh, h0, Wo, bo, activation):
        self.f = activation

        # redundant - see how you can improve it
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]

        thX = T.ivector('X')
        thY = T.iscalar('Y')

        def recurrence(x_t, h_t1):
            # returns h(t), y(t)
            h_t = self.f(self.Wx[x_t] + h_t1.dot(self.Wh) + self.bh)
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            return h_t, y_t

        [h, y], _ = theano.scan(
            fn=recurrence,
            outputs_info=[self.h0, None],
            sequences=thX,
            n_steps=thX.shape[0],
        )

        py_x = y[-1, 0, :]  # only interested in the final classification of the sequence
        prediction = T.argmax(py_x)
        self.predict_op = theano.function(
            inputs=[thX],
            outputs=prediction,
            allow_input_downcast=True,
        )
        return thX, thY, py_x, prediction


def train_poetry():
    X, Y, V = myutil.get_poetry_classifier_data(samples_per_class=500)
    rnn = RNN_class(2, V)
    rnn.fit(X, Y, learning_rate=10e-7, show_fig=True, activation=T.nnet.relu, epochs=10)


if __name__ == '__main__':
    train_poetry()
