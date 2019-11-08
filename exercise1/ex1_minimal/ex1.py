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


def hypothesis(theta: np.array, X: np.array) -> np.array:
    y_theta = np.matmul(X, theta)
    return y_theta


def costFunction(y_theta: np.array, y: np.array) -> float:
    cost = np.sum(((y_theta - y)**2), axis=0) / (2*len(y))
    return cost


def gradientDescent(alpha: float, theta: np.array, y_theta: np.array,
                    X: np.array, y: np.array) -> np.array:
    theta_correction = (alpha/len(y)) * np.matmul(X.transpose(), (y_theta - y))

    theta_updated = theta - theta_correction

    return theta_updated


def plotCostFunction(iteration: List[int], J: List[float],
                     filename: str) -> None:
    plt.figure()
    plt.plot(iteration, J)
    plt.xlim([-10, max(iteration)])
    plt.ylim([min(J)*0.95, max(J)*1.05])
    plt.xlabel('Iteration')
    plt.ylabel(r'$J(\theta) = \frac{1}{2m} \sum\limits^m_{i=1}'
               r'\big(h_{\theta}(x^{(i)})-y^{(i)}\big)^2$')
    plt.title('Cost function')
    plt.savefig(filename + '.png', dpi=150)
    plt.close()


def plotLinearRegOneVar(X: np.array, y: np.array, modelOutput: np.array,
                        filename: str) -> None:
    plt.figure()
    plt.scatter(X[:, 1], y, marker='+', color='red', label='Training data')
    plt.plot(X[:, 1], modelOutput, label='Linear regression')
    plt.xlabel('Population of city in 10,000s')
    plt.ylabel(r'Profit in \$10,000s')
    plt.title('Linear regression with one variable')
    plt.legend(loc='best')
    plt.savefig(filename + '.png', dpi=150)
    plt.close()


def header(argv: List[str]) -> (float, np.array, np.array):
    if not checkArgvOk(argv):
        sys.exit('ERROR - The arguments passed to "ex1.py" are incorrect. '
                 'Execution format:\n'
                 'python --alpha <learningRate> --file <filename>')
    alpha = float(argv[2])
    data = np.loadtxt(argv[4], delimiter=",")
    X = np.ones(data.shape)
    X[:, 1:] = data[:, :-1]
    y = data[:, -1]

    return alpha, X, y


def body(alpha: float, X: np.array, y: np.array) -> None:
    theta = np.zeros(X.shape[1])

    iteration = []
    J = []
    for i in range(1500):
        y_theta = hypothesis(theta, X)
        cost = costFunction(y_theta, y)
        theta = gradientDescent(0.01, theta, y_theta, X, y)
        iteration.append(i)
        J.append(cost)

    plotCostFunction(iteration, J, 'linearRegression1Variable_costFunction')

    plotLinearRegOneVar(X, y, y_theta, 'linearRegression1Variable_result')


if __name__ == "__main__":
    alpha, X, y = header(sys.argv)
    body(alpha, X, y)
