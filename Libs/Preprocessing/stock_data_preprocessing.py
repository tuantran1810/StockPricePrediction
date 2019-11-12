import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   
import stock_data_feature_generator as FG

def prepairFigure(nRows, nCols, title):
    fig, axes = plt.subplots(nRows, nCols, figsize=(10, 10))
    # fig.tight_layout()
    plt.subplots_adjust(left=0.1, top=0.9, right=0.9, hspace=0.4)
    plt.suptitle(title, fontsize=24)
    return (fig, axes)

class DataLoader():
    def __init__(self, path, postProcessing):
        self.stockData = pd.read_csv(path, header = 0)
        self.stockData = self.__processMissingValues()
        postProcessing(self)

    def __processMissingValues(self):
        nullCnt = self.stockData.isna()
        if nullCnt.sum() is not 0:
            self.stockData = self.stockData.fillna(method = "ffill", axis=0).fillna("0")
        return self.stockData

    def dropColumns(self, colsName):
        for name in colsName:
            self.stockData.drop(labels = name, axis = 1, inplace = True)
        return self.stockData

    def renameColumns(self, names):
        self.stockData.columns = names
        return self.stockData

    def getColumn(self, name):
        return self.stockData[name]

    def getDataTable(self):
        return self.stockData

    def enrichReturnData(self):
        self.stockData = FG.return_features(self.stockData)
        return self

    def printOutlier(self, column):
        df_outliers = self.stockData.loc[:,column]
        df_smallest = df_outliers.sort_values(ascending=True)
        df_largest = df_outliers.sort_values(ascending=False)
        print(f"Smallest {column}:")
        print(df_smallest.iloc[:5])
        print(f"Largest {column}:")
        print(df_largest.iloc[:5])

    def plotPriceVsReturn(self):
        fig, axes = prepairFigure(2, 1, "MSFT")
        ax0 = axes[0]
        ax0.set_title("Price")
        ax0.set_ylabel("$")
        self.stockData[["date", "close"]].plot(x = "date", kind = "line", ax = ax0)
        ax1 = axes[1]
        ax1.set_title("Return")
        self.stockData[["date", "return"]].plot(x = "date", kind = "line", ax = ax1)
        plt.show()

class FeatureEnricher():
    def __init__(self, dataTable):
        self.dataTable = dataTable
        self.X = None
        self.y = None

    def addTrendFeatures(self):
        self.dataTable = FG.macd(self.dataTable)
        self.dataTable = FG.ma(self.dataTable)
        self.dataTable = FG.parabolic_sar(self.dataTable)
        return self

    def plotTrendFeatures(self):
        fig, axes = prepairFigure(3, 1, "MSFT - Trend Features")
        ax = axes[0]
        ax.set_title("MACD")
        self.dataTable[["date", "macd_line", "macd_9_day"]].plot(x="date", kind="line", ax=ax, secondary_y=False)
        ax = axes[1]
        ax.set_title("MA")
        self.dataTable[["date", "close"]].plot(x="date", kind="line", ax=ax, secondary_y=False)
        self.dataTable[["date", "ma_50_day"]].plot(x="date", kind="line", ax=ax, secondary_y=False)
        self.dataTable[["date", "ma_200_day"]].plot(x="date", kind="line", ax=ax, secondary_y=False)
        self.dataTable[["date", "ma_50_200"]].plot(x="date", kind="line", ax=ax, secondary_y=True)
        ax = axes[2]
        ax.set_title("SAR")
        self.dataTable[["date", "close"]].plot(x="date", kind="line", ax=ax, secondary_y=False)
        self.dataTable[["date", "sar"]].plot(x="date", style=".", ax=ax, secondary_y=False)
        plt.show()

    def addMomentumFeatures(self):
        self.dataTable = FG.stochastic_oscillator(self.dataTable)
        self.dataTable = FG.commodity_channel_index(self.dataTable)
        self.dataTable = FG.rsi(self.dataTable)
        return self

    def plotMomentumFeatures(self):
        fig, axes = prepairFigure(3, 1, "MSFT - Momentum Features")
        ax = axes[0]
        ax.set_title("Stochastic Oscillator")
        self.dataTable.tail(300)[["date", "close"]].plot(x="date", kind="line", ax=ax, secondary_y=False)
        self.dataTable.tail(300)[["date", "stochastic_oscillator"]].plot(x="date", kind="line", ax=ax, secondary_y=True)
        ax = axes[1]
        ax.set_title("Commodity Channel Index")
        self.dataTable.tail(300)[["date", "close"]].plot(x="date", kind="line", ax=ax, secondary_y=False)
        self.dataTable.tail(300)[["date", "cci"]].plot(x="date", kind="line", ax=ax, secondary_y=True)
        ax = axes[2]
        ax.set_title("RSI")
        self.dataTable.tail(300)[["date", "close"]].plot(x="date", kind="line", ax=ax, secondary_y=False)
        self.dataTable.tail(300)[["date", "rsi"]].plot(x="date", kind="line", ax=ax, secondary_y=True)
        plt.show()

    def addVolatilityFeatures(self):
        self.dataTable["5d_volatility"] = self.dataTable["return"].rolling(5).std()
        self.dataTable["21d_volatility"] = self.dataTable["return"].rolling(21).std()
        self.dataTable["60d_volatility"] = self.dataTable["return"].rolling(60).std()
        self.dataTable = FG.bollinger_bands(self.dataTable)
        self.dataTable = FG.average_true_range(self.dataTable)
        return self

    def plotVolatilityFeatures(self):
        fig, axes = prepairFigure(2, 1, "MSFT - Volatility Features")
        ax = axes[0]
        ax.set_title("Bollinger")
        self.dataTable.tail(100)[["date", "close"]].plot(x="date", kind="line", ax=ax, secondary_y=False)
        self.dataTable.tail(100)[["date", "bollinger"]].plot(x="date", kind="line", ax=ax, secondary_y=True)
        ax = axes[1]
        ax.set_title("Average True Range")
        self.dataTable.tail(100)[["date", "close"]].plot(x="date", kind="line", ax=ax, secondary_y=False)
        self.dataTable.tail(100)[["date", "atr"]].plot(x="date", kind="line", ax=ax, secondary_y=True)

    def addVolumeFeatures(self):
        self.dataTable["volume_rolling"] = self.dataTable["volume"] / self.dataTable["volume"].shift(21)
        self.dataTable = FG.on_balance_volume(self.dataTable)
        self.dataTable = FG.chaikin_oscillator(self.dataTable)
        return self

    def plotVolumeFeatures(self):
        fig, axes = prepairFigure(2, 1, "MSFT - Volume Features")
        ax = axes[0]
        ax.set_title("On Balance Volume")
        self.dataTable.tail(100)[["date", "close"]].plot(x="date", kind="line", ax=ax, secondary_y=False)
        self.dataTable.tail(100)[["date", "on_balance_volume"]].plot(x="date", kind="line", ax=ax, secondary_y=True)
        ax = axes[1]
        ax.set_title("Chaikin Oscillator")
        self.dataTable.tail(100)[["date", "close"]].plot(x="date", kind="line", ax=ax, secondary_y=False)
        self.dataTable.tail(100)[["date", "chaikin_oscillator"]].plot(x="date", kind="line", ax=ax, secondary_y=True)

    def generateFinalData(self):
        self.X = self.dataTable.loc[200:len(self.dataTable)-1, ["return", "close_to_open", "close_to_high", "close_to_low",
                    "macd_diff", "ma_50_200", "sar", "stochastic_oscillator",
                    "cci", "rsi", "5d_volatility", "21d_volatility", "60d_volatility",
                    "bollinger", "atr", "on_balance_volume", "chaikin_oscillator"]]
        self.y = self.dataTable.loc[200:len(self.dataTable)-1, ["y"]]
        return self

    def getXy(self):
        return self.X, self.y
