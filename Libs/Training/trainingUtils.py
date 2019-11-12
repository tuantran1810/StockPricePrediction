from sklearn import linear_model
from sklearn import tree
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import pandas as pd

def _performance(target, predict):
    return (mean_absolute_error(target, predict), mean_squared_error(target, predict), r2_score(target, predict))

def _linearRegressionTrain(Xtrain, ytrain):
    model = linear_model.LinearRegression()
    polymodel = PolynomialFeatures(degree=2).fit(Xtrain)
    Xtrain = polymodel.transform(Xtrain)
    model.fit(Xtrain, ytrain)
    return model, polymodel

def _getTrainingData(Xlearn, ylearn, tscv):
    for trainIndex, validateIndex in tscv:
        Xtrain = Xlearn[trainIndex,:]
        ytrain = ylearn[trainIndex,:]
        Xtest = Xlearn[validateIndex,:]
        ytest = ylearn[validateIndex,:]
        yield trainIndex[-1], Xtrain, ytrain, Xtest, ytest

def _decisionTreeTrain(Xtrain, ytrain):
    model = tree.DecisionTreeRegressor(splitter="best", max_depth=12,
        min_samples_split=5, max_leaf_nodes=100, random_state=1)
    polymodel = PolynomialFeatures(degree=2).fit(Xtrain)
    Xtrain = polymodel.transform(Xtrain)
    model.fit(Xtrain, ytrain)
    return model, polymodel

def _decisionTreeGridSearch(Xtrain, ytrain, tscv):
    Xtrain = PolynomialFeatures(degree=2).fit_transform(Xtrain)
    model = tree.DecisionTreeRegressor(splitter="best", max_depth=14,
        min_samples_split=5, max_leaf_nodes=None, random_state=2)
    splitter = ["best", "random"]
    max_depth = [12, 14, 16, 18, 20]
    min_samples_split = [5, 10, 15, 20]
    max_leaf_nodes = [None, 50, 100, 150, 200]
    params = dict(splitter=splitter, max_depth=max_depth,
        min_samples_split=min_samples_split, max_leaf_nodes=max_leaf_nodes)
    grid = GridSearchCV(estimator=model, param_grid=params, cv=tscv)
    grid_result = grid.fit(Xtrain, ytrain)
    print("Best score: {} using {}".format(grid_result.best_score_, grid_result.best_params_))

def _neuralNetworkTrain(Xtrain, ytrain):
    import warnings
    warnings.filterwarnings('ignore')
    polymodel = PolynomialFeatures(degree=2).fit(Xtrain)
    model = MLPRegressor(hidden_layer_sizes=(250, 200), solver="adam",
        activation="relu", random_state=2)
    Xtrain = polymodel.transform(Xtrain)
    model.fit(Xtrain, ytrain)
    return model, polymodel

def _neuralNetworkGridSearch(Xtrain, ytrain, tscv):
    import warnings
    warnings.filterwarnings('ignore')
    Xtrain = PolynomialFeatures(degree=2).fit_transform(Xtrain)
    model = MLPRegressor(hidden_layer_sizes=(200, 200), solver="adam",
        activation="relu", random_state=2)
    hidden_layer_sizes = [(50, 50), (100, 50), (50, 100), (100, 100),
                            (100, 150), (150, 100), (150, 150), (150, 200),
                            (200, 150), (200, 200), (200, 250), (250, 200), (250, 250)]
    params = dict(hidden_layer_sizes=hidden_layer_sizes)
    grid = GridSearchCV(estimator=model, param_grid=params, cv=tscv)
    grid_result = grid.fit(Xtrain, ytrain)
    print("Best score: {} using {}".format(grid_result.best_score_, grid_result.best_params_))

def _plotTrainResult(result):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    plt.suptitle("Training Result", fontsize=24)
    plt.subplots_adjust(left=0.1, top=0.9, right=0.9, bottom=0.1, hspace=0.6)

    axes[0].set_title("MAE")
    axes[0].set_xlabel("Final train index of fold")
    result[["Train Index", "MAE train", "MAE test"]].plot(x="Train Index", kind="bar", ax=axes[0], logy=True)

    axes[1].set_title("MSE")
    axes[1].set_xlabel("Final train index of fold")
    result[["Train Index", "MSE train", "MSE test"]].plot(x="Train Index", kind="bar", ax=axes[1], logy=True)

    axes[2].set_title("R2")
    axes[2].set_xlabel("Final train index of fold")
    axes[2].set_ylim(0, 1)
    result[["Train Index", "R2 train", "R2 test"]].plot(x="Train Index", kind="bar", ax=axes[2])

def _predictModel(Xtest, ytest, model, polymodel):
    Xtest = polymodel.transform(Xtest)
    ypredict = model.predict(Xtest)
    maeTest, mseTest, r2Test = _performance(ytest, ypredict)
    return maeTest, mseTest, r2Test, ypredict

def _plotTestResult(maeTest, mseTest, r2Test, ytest, ypredict):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    plt.suptitle("Testing Result", fontsize=24)
    plt.subplots_adjust(left=0.1, top=0.9, right=0.9, bottom=0.1, hspace=0.6)
    axes[0].set_title("MAE - MSE - R2")
    axes[0].bar([0, 1, 2], [maeTest, mseTest, r2Test], align='center', alpha=1, tick_label = ["MAE","MSE", "R2"], log = True)

    axes[1].set_title("Prediction vs Reality")
    axes[1].plot(ytest, label = "reality")
    axes[1].plot(ypredict, label = "prediction")
    axes[1].legend()
    return axes
