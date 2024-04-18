import numpy as np
from math import sqrt
class PolynomialRegression():

    def __init__(self, degree):
        # Data initialization
        self.CorrelationCoefficent = 0
        self.DeterminationCoefficent = 0
        self.degree = degree
        self.predictorX = (108,115,106,97,95,91,97,83,83,78,54,67,56,53,61,115,81,78,30,45,99,32,25,28,90,89)
        self.responseY  = (95,96,95,97,93,94,95,93,92,86,73,80,65,69,77,96,87,89,60,63,95,61,55,56,94,93)

    def initXMatrix(self):
        # Initialize design matrix X
        XMatrix = np.ones((len(self.predictorX), self.degree + 1))
        for i in range(self.degree + 1):
            XMatrix[:, i] *= (np.array(self.predictorX) ** i)
        return XMatrix
    
    def initYMatrix(self):
        # Initialize response matrix Y
        return np.array(self.responseY).reshape(-1,1)
    
    def BMatrix(self):
        #Init matrixes
        X = SSBBCase.initXMatrix()
        Y = SSBBCase.initYMatrix()

        # Calculate coefficients matrix B using equation B = (X^t * X)^-1 * X^t * Y
        X_t = np.transpose(X)
        Y_t = np.transpose(Y)

        # X * Y^t X_columns != Y_rows.  Equation explained in class
        #print(X.shape)
        #print(Y_t.shape)

        # Calculate X^t * Y.            Right equation
        X_t_Y = np.dot(X_t, Y)

        # Calculate X^t * X
        X_t_X = np.dot(X_t, X)
        # Calculate the inverse of X^t * X
        X_t_X_inv = np.linalg.inv(X_t_X)

        # Calculate matrix B
        B = np.dot(X_t_X_inv, X_t_Y)
        return B
    
    # Coefficient of determination (R-squared)
    def r_squared(self, Y_pred):
        Y = self.responseY
        # Calculate total sum of squares (SST)
        SST = np.sum((Y - np.mean(Y))**2)
        # Calculate sum of squares of residuals (SSE)
        SSE = np.sum((Y - Y_pred)**2)
        # Calculate R^2
        R_squared = 1 - (SSE / SST)
        return R_squared
    
    # Correlation coefficient
    def correlation_coefficient(self, r_2):
        return sqrt(r_2)
    
    # Predictions for polynomial regression  
    def Predict(self, predictionValues):
        predictionValues = np.array(predictionValues)
        self.PredictionValues = 0
        self.PredictionValues = predictionValues

        self.Params = self.BMatrix()
        predictions = 0

        for i in range(len(self.Params)):
            predictions = predictions + self.Params[i] * predictionValues ** i

        self.Predictions = np.array(predictions)

        return self.Predictions

    def CalculateCoefficents(self):
        linearYPred = self.Predict(self.predictorX)
        self.CorrelationCoefficent = self.correlation_coefficient(SSBBCase.r_squared(linearYPred))
        self.DeterminationCoefficent = self.r_squared(linearYPred)

    
    def PrintResults(self):
        print("\nRegression equation")
        print("\nMachine Efficiency =", end="")

        for i in range(len(self.Params)):
            print(" " +' '.join(map(str, self.Params[i])), end="")
            if(i == 1):
                print(" Batch Size", end=" ")
            elif(i > 1):
                print(f" Batch Size^{i} ", end="+")

        print(f"\nCorrelation coefficient = {self.CorrelationCoefficent}")
        print(f"Determination coefficient = {self.DeterminationCoefficent}")
        
        
# Values to predict
predictionValues = (123,15,60)
predictionValuesKnown = (108, 115,106)

# Regression
SSBBCase = PolynomialRegression(3)
knownPred = SSBBCase.Predict(predictionValuesKnown)
unknownPred = SSBBCase.Predict(predictionValues)
SSBBCase.CalculateCoefficents()
SSBBCase.PrintResults()

print("\nPredictions for known values")
for i in range(len(knownPred)):
    print(f"Batch Size = {SSBBCase.predictorX[i]}, Machine Efficiency Predicted = {knownPred[i]}, Machine Efficiency = {SSBBCase.responseY[i]}")

print("\nPredictions for unknown values")
for i in range(len(unknownPred)):
    print(f"Batch Size = {predictionValues[i]}, Machine Efficiency = {unknownPred[i]}")