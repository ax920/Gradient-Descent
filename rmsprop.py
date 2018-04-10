import numpy as np
import random
import sklearn
from sklearn.datasets.samples_generator import make_regression
import pylab
from scipy import stats


def rmsprop(alpha, x, y, ep, max_iter=10000, decay = 0.9, eps = 1e-8):
    converged = False
    iter = 0
    m = x.shape[0]  # number of samples
    loss = []
    count = []
    # initial theta
    t0 = 0
    t1 = 0
    RMSnew0 = 0
    RMSnew1 = 0
    # total error, J(theta)
    J = sum([(t0 + t1 * x[i] - y[i]) ** 2 for i in range(m)])

    # Iterate Loop
    while not converged:
        # for each training sample, compute the gradient (d/d_theta j(theta))
        grad0 = 1.0 / m * sum([(t0 + t1 * x[i] - y[i]) for i in range(m)])
        grad1 = 1.0 / m * sum([(t0 + t1 * x[i] - y[i]) * x[i] for i in range(m)])
        RMSnew0 = decay*RMSnew0 + (1-decay)*(grad0**2)
        RMSnew1 = decay*RMSnew1 + (1-decay)*(grad1**2)

        t0 = t0 - (grad0/(RMSnew0**0.5 + eps))*alpha
        t1 = t1 - (grad1/(RMSnew1**0.5 + eps))*alpha

        # mean squared error
        e = sum([(t0 + t1 * x[i] - y[i]) ** 2 for i in range(m)])

        if abs(J - e) <= ep:
            print('Converged, iterations: ', iter)
            converged = True
        loss.append(e)
        J = e  # update error
        count.append(iter)
        iter += 1  # update iter

        if iter == max_iter:
            print('Max iterations exceeded!')
            converged = True

    return t0, t1, count, loss


if __name__ == '__main__':

    x, y = make_regression(n_samples=1000, n_features=1, n_informative=1,
                           random_state=0, noise=100)
    print
    'x.shape = %s y.shape = %s' % (x.shape, y.shape)


    alpha = 0.01  # learning rate
    ep = 0.01  # convergence criteria

    # call gredient decent, and get intercept(=theta0) and slope(=theta1)
    theta0, theta1, count1, loss1 = rmsprop(alpha, x, y, ep, max_iter=10000,decay=0.9, eps = 1e-8)
    print("theta0 = ", theta0, "theta1 = ", theta1)

    # check with scipy linear regression
    slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x[:, 0], y)
    print("intercept = ", intercept, "slope = ", slope)

    # plot
    for i in range(x.shape[0]):
        y_predict = theta0 + theta1 * x

    pylab.plot(x, y, 'o')
    pylab.plot(x, y_predict, 'k-')
    pylab.show()
    pylab.plot()
    print
    "Done!"
    pylab.plot(count1,loss1)
    pylab.show()