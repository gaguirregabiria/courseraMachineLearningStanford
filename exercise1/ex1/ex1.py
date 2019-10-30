import sys
import os.path
from typing import List
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('text', usetex=True)
plt.rc('font', family='serif')


def checkArgvOk(argv: List[str]) -> bool:
    if(argv[1] == '--file' and os.path.isfile(argv[2])):
        argv_ok = True
    elif not os.path.isfile(argv[2]):
        sys.exit('ERROR - File ' + argv[2] + ' does not exist')
    else:
        argv_ok = False

    return argv_ok


def hypothesis(theta: pd.DataFrame, x: pd.DataFrame) -> np.array:
    y_theta = (x.multiply(theta.values, axis='columns')).sum(axis='columns')
    y_theta = pd.DataFrame(y_theta, columns=['y'])
    return y_theta


def costFunction(y_theta: pd.DataFrame, y: pd.DataFrame) -> float:
    cost = ((y_theta - y)**2).sum(axis='rows') / (2*len(y))
    return cost.values[0]


def gradientDescent(theta: pd.DataFrame, y_theta: pd.DataFrame,
                    x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    gd_summatory = pd.DataFrame(x.values*(y_theta - y).values,
                                columns=theta.columns)
    theta_correction = pd.DataFrame(
        (0.01/len(y)) * gd_summatory.sum(axis='rows')).transpose()

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


def plotLinearRegOneVar(data: pd.DataFrame, modelOutput: pd.DataFrame,
                        filename: str) -> None:
    plt.figure()
    plt.scatter(data['x1'].values, data['y'].values, marker='+', color='red',
                label='Training data')
    plt.plot(data['x1'].values, modelOutput.values, label='Linear regression')
    plt.xlabel('Population of city in 10,000s')
    plt.ylabel(r'Profit in \$10,000s')
    plt.title('Linear regression with one variable')
    plt.legend(loc='best')
    plt.savefig(filename + '.png', dpi=150)
    plt.close()


def header(argv: List[str]) -> pd.DataFrame:
    if not checkArgvOk(argv):
        sys.exit('ERROR - The arguments passed to "ex1.py" are incorrect. '
                 'Execution format:\n', 'python --file <filename>')

    data = pd.read_csv(argv[2], sep=",", header=None)
    data.insert(loc=0, column='x0', value=1)
    data.columns = ['x0', 'x1', 'y']

    return data


def body(data: pd.DataFrame) -> None:
    theta = pd.DataFrame(data=np.zeros([1, len(data.columns)-1]),
                         columns=['t0', 't1'])

    iteration = []
    J = []
    for i in range(1500):
        y_theta = hypothesis(theta, data.drop('y', axis=1))
        cost = costFunction(y_theta, data[['y']])
        theta = gradientDescent(theta, y_theta, data.drop('y', axis=1),
                                data[['y']])
        iteration.append(i)
        J.append(cost)

    plotCostFunction(iteration, J, 'linearRegression1Variable_costFunction')

    plotLinearRegOneVar(data, y_theta, 'linearRegression1Variable_result')


if __name__ == "__main__":
    data = header(sys.argv)
    body(data)
