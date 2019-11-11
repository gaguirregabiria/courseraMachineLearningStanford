import sys
import os.path
from typing import List
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
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


def plotLinearRegOneVar(X: pd.DataFrame, y: pd.DataFrame,
                        modelOutput: np.array, filename: str) -> None:
    plt.figure()
    plt.scatter(X.values, y.values, marker='+', color='red',
                label='Training data')
    plt.plot(X.values, modelOutput, label='Linear regression')
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
    column_names = ['x'+str(i) for i in range(1, data.shape[1])]
    column_names.append('y')
    data.columns = column_names

    return data


def body(data: pd.DataFrame) -> None:
    X = data.filter(regex='^x', axis=1)
    y = data['y']

    stdScaler = StandardScaler()
    stdScaler.fit(X)
    Xn = stdScaler.transform(X)
    linReg = LinearRegression(fit_intercept=True, normalize=False, copy_X=True)
    linReg.fit(Xn, y)
    print('R^2 score of the regression:', linReg.score(Xn, y))
    y_prediction = linReg.predict(Xn)

    plotLinearRegOneVar(X, y, y_prediction, 'linearRegression1Variable_result')


if __name__ == "__main__":
    data = header(sys.argv)
    body(data)
