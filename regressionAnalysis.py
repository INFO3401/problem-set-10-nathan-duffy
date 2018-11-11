from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
import csv
import numpy as np
import pandas as pd

class analysisData:


    def __init__(self, filename):
        self.variables = []
        self.filename = filename

    def parseFile(self):
        self.dataset = pd.read_csv(self.filename)
        self.variables = self.dataset.columns

dataParser = analysisData("./candy-data.csv")
dataParser.parseFile()

class linearAnalysis(object):

    def __init__(self, targetY):
        self.bestX = None
        self.targetY = targetY
        self.fit = None

    def runSimpleAnalysis(self, dataParser):

        dataset = dataParser.dataset

        best_pred = 0
        for column in dataParser.variables:

            if column == self.targetY or column == 'competitorname':
                continue

            x_values = dataset[column].values.reshape(-1,1)
            y_values = dataset[self.targetY].values

            regr = LinearRegression()
            regr.fit(x_values, y_values)
            preds = regr.predict(x_values)
            score = r2_score(y_values, preds)

            if score > best_pred:
                best_pred = score
                self.bestX = column

        self.fit = best_pred
        print(self.bestX)
        print(self.fit)
        print(regr.coef_)
        print(regr.intercept_)

class logisticAnalysis(object):

    def __init__(self, targetY):
        self.bestX = None
        self.targetY = targetY
        self.fit = None

    def runSimpleAnalysis(self, dataParser):

        dataset = dataParser.dataset

        best_pred = 0
        for column in dataParser.variables:

            if column == self.targetY or column == 'competitorname':
                continue

            x_values = dataset[column].values.reshape(-1,1)
            y_values = dataset[self.targetY].values

            regr = LogisticRegression()
            regr.fit(x_values, y_values)
            preds = regr.predict(x_values)
            score = r2_score(y_values, preds)

            if score > best_pred:
                best_pred = score
                self.bestX = column

        self.fit = best_pred
        print(self.bestX)
        print(self.fit)
        print(regr.coef_)
        print(regr.intercept_)

    def runMultipleRegression(self, dataParser):

        dataset = dataParser.dataset
        clean_dataset = dataset.drop([self.targetY, 'competitorname', 'chocolate'], axis = 1)
        x_values = clean_dataset.values
        y_values = dataset[self.targetY].values

        regr = LogisticRegression()
        regr.fit(x_values, y_values)
        preds = regr.predict(x_values)
        score = r2_score(y_values,preds)

        print(clean_dataset.columns)
        print(score)
        print(regr.coef_)
        print(regr.intercept_[0])




#linear_analysis = linearAnalysis(targetY = 'sugarpercent')
#linear_analysis.runSimpleAnalysis(dataParser)

#logistic_analysis = logisticAnalysis(targetY = 'chocolate')
#logistic_analysis.runSimpleAnalysis(dataParser)

multiple_regression = logisticAnalysis(targetY = 'chocolate')
multiple_regression.runMultipleRegression(dataParser)

# Problem 3

# Linear Regression Equation: y = 0.00440378x + 0.257063291665
# Logistic Regression Equation: 1 / 1 + e^-(0.05901723x - 3.08798586)
# Multiple Regression Equation: -2.52858047x1 - 0.19697876x2 + 0.03940308x3 - 0.16539952x4 + 0.49783674x5
# - 0.47591613x6 + 0.81511886x7 - 0.59971553x8 - 0.2581028x9 + 0.3224988x10 + 0.05387906x11 - 1.6826055313

# Problem 4

#a)Independent variable - candy (chocolate vs. caramel)(discrete)
#Dependent variable - sugar content (continuous)
#Null Hypothesis - Chocolate and caramel candies have the same sugar content.

#b)Independent Variable - political affiliation of states (blue vs. red)(discrete)
#Dependent variable - split ticket voters (discrete)
#Null Hypothesis - There are an equal amount of split ticket voters in blue and red states.

#c)Independent variable - battery life (continuous)
#Dependent variable - sell rate (continuous)
#Null Hypothesis - The phones sell at the same rate regardless of battery life.
