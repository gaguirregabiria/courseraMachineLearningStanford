import sys
from typing import List
import numpy as np
from scipy.optimize import minimize
from ex2_functions import checkArgvOk, plotLogisticRegression, sigmoid,\
    checkSigmoid, hypothesis, costFunction, featureNormalization,\
    checkCostFunction, gradientDescent, plotCostFunction,\
    calculateLogisticRegresionAreas


opt_iter = 0
opt_progress = []


def callbackFunction(theta: np.array) -> None:
    global opt_iter
    global opt_progress
    opt_progress.append([opt_iter, *theta])
    opt_iter += 1


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
    # Import global variables for optimization tracing
    global opt_iter
    global opt_progress

    # Initialize LogReg parameters (theta) to a vector of zeros
    theta = np.zeros(X.shape[1])

    # Visualize the data
    plotLogisticRegression(X[:, 1:], y, None,
                           'logisticRegression_data1Visualization')

    # Check that Sigmoid and cost function are well implemented
    checkSigmoid()
    checkCostFunction(X, y)

    # Feature normalization
    Xn, mean, std = featureNormalization(X)

    # Obtain theta with custom implementation of Gradient Descent
    iteration = []
    J = []
    for i in range(400):
        cost = costFunction(theta, Xn, y)
        theta = gradientDescent(alpha, theta, Xn, y)
        iteration.append(i)
        J.append(cost)

    # Plot the cost function as it is optimized by GD
    plotCostFunction(iteration, J, alpha,
                     'logisticRegression_data1CostFunction_LR' +
                     str(alpha))

    # Obtain theta with scipy optimization function
    result = minimize(costFunction, np.zeros(X.shape[1]), (Xn, y),
                      callback=callbackFunction)
    if not result.success:
        sys.exit("Optimization function wasn't able to minimize the cost "
                 "function")

    # Plot the cost function as it is optimized by optimize.minimize
    opt_progress = np.asarray(opt_progress)
    opt_cost = [costFunction(t, Xn, y) for t in opt_progress[:, 1:]]
    plotCostFunction(opt_progress[:, 0], opt_cost, -1,
                     'logisticRegression_data1CostFunction_optimization')

    # Check both results are similar to a (0.5% difference allowed)
    try:
        assert(all(abs(1-theta/result.x) < 0.005))
    except AssertionError:
        sys.exit("The results obtained with Gradient Descent and with the "
                 "optimization function differ more than a 0.5%")

    # Check correct prediction for the given example in the exercise
    student = np.array([([1, 45, 85]-mean)/std])
    assert(round(hypothesis(theta, student)[0], 3) == 0.776)

    # Calculate and plot the decision boundary
    decission_grid = calculateLogisticRegresionAreas(theta, X, mean, std)
    decission_grid[2][decission_grid[2] > 0.5] = 1
    decission_grid[2][decission_grid[2] <= 0.5] = 0

    plotLogisticRegression(X[:, 1:], y, decission_grid,
                           'logisticRegression_data1DecisionGrid')


if __name__ == "__main__":
    alpha, X, y, file = header(sys.argv)
    body(alpha, X, y, file)
