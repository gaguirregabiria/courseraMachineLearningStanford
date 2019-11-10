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
    y_theta = np.matmul(X, theta)
    return y_theta


def costFunction(y_theta: np.array, y: np.array) -> float:
    # cost = np.sum(((y_theta - y)**2), axis=0) / (2*len(y))
    cost = np.matmul((y_theta - y).transpose(), (y_theta - y)) / (2*len(y))
    return cost


def gradientDescent(alpha: float, theta: np.array, y_theta: np.array,
                    X: np.array, y: np.array) -> np.array:
    theta_correction = (alpha/len(y)) * np.matmul(X.transpose(), (y_theta - y))

    theta_updated = theta - theta_correction

    return theta_updated


def plotCostFunction(iteration: List[int], J: List[float], alpha: float,
                     filename: str) -> None:
    plt.figure()
    plt.plot(iteration, J)
    plt.xlim([-10, max(iteration)])
    plt.ylim([min(J)*0.95, max(J)*1.05])
    plt.xlabel('Iteration')
    plt.ylabel(r'$J(\theta) = \frac{1}{2m} \sum\limits^m_{i=1}'
               r'\big(h_{\theta}(x^{(i)})-y^{(i)}\big)^2$')
    plt.suptitle('Cost function')
    plt.title('Learning rate = ' + str(alpha))
    plt.savefig(filename + '.png', dpi=150)
    plt.close()


def plotLinearRegOneVar(x: np.array, y: np.array, modelOutput: np.array,
                        alpha: float, filename: str) -> None:
    plt.figure()
    plt.scatter(x, y, marker='+', color='red', label='Training data')
    plt.plot(x, modelOutput, label='Linear regression')
    plt.xlabel('Population of city in 10,000s')
    plt.ylabel(r'Profit in \$10,000s')
    plt.suptitle('Linear regression with one variable')
    plt.title('Learning rate = ' + str(alpha))
    plt.legend(loc='best')
    plt.savefig(filename + '.png', dpi=150)
    plt.close()