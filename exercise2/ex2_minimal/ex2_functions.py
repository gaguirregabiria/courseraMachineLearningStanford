import sys
import os.path
from typing import List
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('text', usetex=True)
plt.rc('font', family='serif')


def isNumber(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def checkArgvOk(argv: List[str]) -> bool:
    if(len(argv) != 5):
        argv_ok = False
    elif(argv[1] == '--alpha' and isNumber(argv[2]) and argv[3] == '--file' and
         os.path.isfile(argv[4])):
        argv_ok = True
    elif(not isNumber(argv[2])):
        sys.exit('ERROR - learning rate "' + argv[2] + '" is not a number')
    elif not os.path.isfile(argv[4]):
        sys.exit('ERROR - File "' + argv[2] + '" does not exist')
    else:
        argv_ok = False

    return argv_ok


def featureNormalization(X: np.array) -> (np.array, np.array, np.array):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    mean[0] = 0
    std[0] = 1

    Xn = (X - mean)/std
    return Xn, mean, std


def hypothesis(theta: np.array, X: np.array) -> np.array:
    y_theta = sigmoid(np.matmul(X, theta))
    return y_theta


def sigmoid(z: np.array) -> np.array:
    result = 1 / (1 + np.exp(-z))
    return result


def checkSigmoid() -> None:
    assert(sigmoid(np.array([0])) == np.array([0.5]))
    good_result = np.array([1/(1 + np.exp(1)), 0.5, 1/(1 + np.exp(-1))])
    assert(np.array_equal(sigmoid(np.array([-1, 0, 1])), good_result))
    good_result = np.array([[1/(1 + np.exp(100)), 1/(1 + np.exp(500))],
                            [1/(1 + np.exp(-100)), 1/(1 + np.exp(-500))]])
    assert(np.array_equal(sigmoid(np.array([[-100, -500], [100, 500]])),
                          good_result))


def checkCostFunction(X: np.array, y: np.array) -> None:
    theta = np.zeros(X.shape[1])

    Xn, mean, std = featureNormalization(X)
    cost = costFunction(theta, Xn, y)

    assert(round(cost, 3) == 0.693)


def costFunction(theta: np.array, X: np.array, y: np.array) -> float:
    y_theta = hypothesis(theta, X)
    to_sum = -y*np.log(y_theta) - (1 - y)*np.log(1 - y_theta)
    cost = np.sum(to_sum) / len(y)
    return cost


def gradientDescent(alpha: float, theta: np.array, X: np.array,
                    y: np.array) -> np.array:
    y_theta = hypothesis(theta, X)
    theta_correction = (alpha/len(y)) * np.matmul(X.transpose(), (y_theta - y))

    theta_updated = theta - theta_correction

    return theta_updated


def calculateLogisticRegresionAreas(theta: np.array, X: np.array,
                                    mean: np.array, std: np.array) ->\
        List[np.array]:
    ex1 = np.arange(min(X[:, 1]), max(X[:, 1]),
                    (max(X[:, 1]) - min(X[:, 1]))/1000)
    ex2 = np.arange(min(X[:, 2]), max(X[:, 2]),
                    (max(X[:, 2]) - min(X[:, 2]))/1000)

    grid_ex1, grid_ex2 = np.meshgrid(ex1, ex2)
    grid_prediction = np.array(grid_ex1.shape)
    data = np.ones([grid_ex1.size, 3])
    data[:, 1] = grid_ex1.flatten()
    data[:, 2] = grid_ex2.flatten()

    data = (data - mean)/std
    grid_prediction = hypothesis(theta, data)
    grid_prediction = grid_prediction.reshape(grid_ex1.shape)

    return [grid_ex1, grid_ex2, grid_prediction]


def plotCostFunction(iteration: List[int], J: List[float], alpha: float,
                     filename: str) -> None:
    plt.figure()
    plt.plot(iteration, J)
    plt.xlim([-10, max(iteration)])
    plt.ylim([min(J)*0.95, max(J)*1.05])
    plt.xlabel('Iteration')
    plt.ylabel(r'$J(\theta) = \frac{1}{m} \sum\limits^m_{i=1}'
               r'\big[-y^{(i)}\log(h_{\theta}(x^{(i)}))'
               r'-(1-y^{(i)})\big(1-\log(h_{\theta}(x^{(i)}))\big)\big]$')
    plt.suptitle('Cost function')
    plt.title('Learning rate = ' + str(alpha))
    plt.savefig(filename + '.png', dpi=150)
    plt.close()


def plotLogisticRegression(x: np.array, y: np.array, decissionGrid: np.array,
                           filename: str) -> None:
    positive_x = x[y == 1, :]
    negative_x = x[y == 0, :]
    plt.figure()
    if decissionGrid is not None:
        plt.contour(decissionGrid[0], decissionGrid[1], decissionGrid[2],
                    colors='blue', linewidths=0.8)
        plt.title('Decission boundary at prediction = 0.5')
    plt.scatter(positive_x[:, 0], positive_x[:, 1], marker='+', color='black',
                label='Admitted')
    plt.scatter(negative_x[:, 0], negative_x[:, 1], marker='o', color='yellow',
                edgecolors='black', label='Not admitted')
    plt.xlim([min(min(positive_x[:, 0]), min(negative_x[:, 0])),
              max(max(positive_x[:, 0]), max(negative_x[:, 0]))])
    plt.ylim([min(min(positive_x[:, 1]), min(negative_x[:, 1])),
              max(max(positive_x[:, 1]), max(negative_x[:, 1]))])
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.suptitle('Logistic regression')
    plt.legend(loc=1)
    plt.savefig(filename + '.png', dpi=150)
    plt.close()
