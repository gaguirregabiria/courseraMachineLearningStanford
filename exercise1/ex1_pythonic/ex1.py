import sys
from typing import List
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from ex1_functions import checkArgvOk, plotLinearRegOneVar


def header(argv: List[str]) -> (pd.DataFrame, str):
    if not checkArgvOk(argv):
        sys.exit('ERROR - The arguments passed to "ex1.py" are incorrect. '
                 'Execution format:\n', 'python --file <filename>')

    data = pd.read_csv(argv[2], sep=",", header=None)
    column_names = ['x'+str(i) for i in range(1, data.shape[1])]
    column_names.append('y')
    data.columns = column_names

    return (data, argv[2])


def body(data: pd.DataFrame, data_file: str) -> None:
    X = data.filter(regex='^x', axis=1)
    y = data['y']

    stdScaler = StandardScaler()
    stdScaler.fit(X)
    Xn = stdScaler.transform(X)
    linReg = LinearRegression(fit_intercept=True, normalize=False, copy_X=True)
    linReg.fit(Xn, y)
    print('R^2 score of the regression:', linReg.score(Xn, y))
    y_prediction = linReg.predict(Xn)

    if 'ex1data1.txt' in file:
        plotLinearRegOneVar(X, y, y_prediction,
                            'linearRegression1Variable_result')
    else:
        house_data = stdScaler.transform([[1650, 3]])
        house_price = linReg.predict(house_data)
        print('The predicted price of a 1650 square feet house with 3',
              'bedrooms is', '{:.2f}'.format(house_price[0]), 'USD')


if __name__ == "__main__":
    data, file = header(sys.argv)
    body(data, file)
