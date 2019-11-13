import sys
import os.path
from typing import List
import numpy as np
import pandas as pd
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
