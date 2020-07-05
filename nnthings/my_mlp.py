import numpy as np
import matplotlib.pyplot as plt

class HyperbolicTangentActivationFunction:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def my_tanh(self, x):
        return self.a * np.tanh(self.b * x)

    def d_my_tanh(self, x):
        return self.a * self.b * (1 - (np.tanh(self.b * x)) ** 2)


class MyMLP:
    def __init__(self, desc, activation_function):
        self.desc = desc
        self.w = []
        self.af = activation_function

    def get_w(self):
        return self.w

    def mlp_generic_training(self, trainning_set, eta, alpha, epochs, should_plot = False):
        desc = self.desc
        af = self.af
        nl = np.size(desc)
        w = []
        w_past = []
        dw = []
        u = []
        g = []
        y = []
        for i in range(0, nl - 1):
            w.append(np.random.rand(desc[i], desc[i + 1]))
            w_past.append(np.zeros([desc[i], desc[i + 1]]))
            dw.append(np.zeros([desc[i], desc[i + 1]]))
            u.append(np.zeros(desc[i + 1]))
            g.append(np.zeros(desc[i + 1]))
            y.append(np.zeros(desc[i + 1]))

        n_samples = np.size(trainning_set, 0)
        evet = []
        enow = 0
        epast = 0

        for m in range(0, epochs):
            trainning_set_input = trainning_set[:, 0:desc[0]]
            trainning_set_result = trainning_set[:, desc[0]:trainning_set.shape[1]]
            for n in range(0, n_samples):

                u[0] = trainning_set_input[n, :].dot(w[0])
                y[0] = af.my_tanh(u[0])

                for i in range(1, nl - 1):
                    u[i] = y[i - 1].dot(w[i])
                    y[i] = af.my_tanh(u[i])

                e = trainning_set_result[n] - y[nl - 2]
                epast = enow
                enow = 0.5 * np.sum(e ** 2)
                evet = np.append(evet, enow)

                g[nl - 2] = af.d_my_tanh(u[nl - 2]) * e

                for i in reversed(range(0, nl - 2)):
                    for j in range(0, desc[i + 1]):
                        g[i][j] = af.d_my_tanh(u[i][j]) * np.sum(g[i + 1] * w[i + 1][j, :])

                dw[0] = alpha * w_past[0] + eta * g[0] * (np.array([trainning_set_input[n, :]]).T * np.ones([desc[0], desc[1]]))

                for i in range(1, nl - 1):
                    dw[i] = alpha * w_past[i] + eta * g[i] * (np.array([y[i - 1]]).T * np.ones([desc[i], desc[i + 1]]))

                for i in range(0, nl - 1):
                    w_past[i] = w[i]

                for i in range(0, nl - 1):
                    w[i] = w[i] + dw[i]

            trainning_set = np.random.permutation(trainning_set)
        self.w = w
        if should_plot:
            plt.plot(evet)
            plt.show()


    def mlp(self, x):
        af = self.af
        ny = np.size(self.w)
        u = []
        y = []

        u.append(x.dot(self.w[0]))
        y.append(af.my_tanh(u[0]))

        for i in range(1, ny):
            u.append(y[i - 1].dot(self.w[i]))
            y.append(af.my_tanh(u[i]))

        return y[ny - 1]
