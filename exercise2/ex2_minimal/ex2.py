import sys
from typing import List
import numpy as np
from ex2_functions import checkArgvOk, plotLogisticRegression, sigmoid,\
    checkSigmoid, hypothesis, costFunction, featureNormalization,\
    checkCostFunction, gradientDescent, plotCostFunction,\
    calculateLogisticRegresionAreas
# featureNormalization, hypothesis,\
    # costFunction, gradientDescent, plotCostFunction, plotLinearRegOneVar


def header(argv: List[str]) -> (np.array, np.array, str):
    if not checkArgvOk(argv):
        sys.exit('ERROR - The arguments passed to "ex1.py" are incorrect. '
                 'Execution format:\n'
                 'python ex2.py --alpha <learningRate> --file <filename>')
    alpha = float(argv[2])
    data = np.loadtxt(argv[4], delimiter=",")
    X = np.ones(data.shape)
    X[:, 1:] = data[:, :-1]
    y = data[:, -1]

    return alpha, X, y, argv[4]


def body(alpha: float, X: np.array, y: np.array, file: str) -> None:
    theta = np.zeros(X.shape[1])

    plotLogisticRegression(X[:, 1:], y, None,
                           'logisticRegresion_data1Visualization')

    checkSigmoid()
    checkCostFunction(X, y)

    iteration = []
    J = []
    for i in range(400):
        Xn, mean, std = featureNormalization(X)
        y_theta = hypothesis(theta, Xn)
        cost = costFunction(y_theta, y)
        theta = gradientDescent(alpha, theta, y_theta, Xn, y)
        iteration.append(i)
        J.append(cost)

    plotCostFunction(iteration, J, alpha,
                     'logisticRegression_data1CostFunction_LR' +
                     str(alpha))

    student = np.array([([1, 45, 85]-mean)/std])
    assert(round(hypothesis(theta, student)[0], 3) == 0.776)

    decission_grid = calculateLogisticRegresionAreas(theta, X, mean, std)
    decission_grid[2][decission_grid[2] > 0.5] = 1
    decission_grid[2][decission_grid[2] <= 0.5] = 0 

    plotLogisticRegression(X[:, 1:], y, decission_grid,
                           'logisticRegresion_data1DecissionGrid')

if __name__ == "__main__":
    alpha, X, y, file = header(sys.argv)
    body(alpha, X, y, file)
