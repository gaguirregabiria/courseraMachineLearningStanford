import sys
from typing import List
import numpy as np
from ex1_functions import checkArgvOk, featureNormalization, hypothesis,\
    costFunction, gradientDescent, plotCostFunction, plotLinearRegOneVar


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
    for i in range(50):
        Xn, mean, std = featureNormalization(X)
        y_theta = hypothesis(theta, Xn)
        cost = costFunction(y_theta, y)
        theta = gradientDescent(alpha, theta, y_theta, Xn, y)
        iteration.append(i)
        J.append(cost)

    plotCostFunction(iteration, J, alpha,
                     'linearRegression1Variable_costFunction_LR' +
                     str(alpha))

    plotLinearRegOneVar(X[:,1], y, y_theta, alpha,
                        'linearRegression1Variable_result_LR' + str(alpha))


if __name__ == "__main__":
    alpha, X, y = header(sys.argv)
    body(alpha, X, y)
