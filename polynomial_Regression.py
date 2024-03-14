import numpy as np
from math import sqrt
class PolynomialRegression():
    def __init__(self):
        # Data initialization
        self.predictorX = (108,115,106,97,95,91,97,83,83,78,54,67,56,53,61,115,81,78,30,45,99,32,25,28,90,89)
        self.responseY = (95,96,95,97,93,94,95,93,92,86,73,80,65,69,77,96,87,89,60,63,95,61,55,56,94,93)

    def initXMatrix(self, degree):
        # Initialize design matrix X
        XMatrix = np.ones((len(self.predictorX), degree + 1))
        for i in range(degree + 1):
            XMatrix[:, i] *= (np.array(self.predictorX) ** i)
        return XMatrix
    
    def initYMatrix(self):
        # Initialize response matrix Y
        return np.array(self.responseY).reshape(-1,1)
    
    def BMatrix(self, X, Y):
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
    def predictions(self, params, predictionValues):
        predictionValues = np.array(predictionValues)
        predictions = 0
        for i in range (len(params)):
            predictions = predictions + params[i] * predictionValues ** i
        return np.array(predictions)
    

# Values to predict
predictionValues = (123,15,60)
SSBBCase = PolynomialRegression()

# Linear Regression
XMatrix = SSBBCase.initXMatrix(1)
YMatrix = SSBBCase.initYMatrix()
BMatrix = SSBBCase.BMatrix(XMatrix, YMatrix)
linearParams = BMatrix
linearYPred = SSBBCase.predictions(linearParams, SSBBCase.predictorX)
print("Linear regression equation")
print(f"Machine Efficiency = {float(linearParams[0])} + {float(linearParams[1])} Batch Size")
print(f"Correlation coefficient = {SSBBCase.correlation_coefficient(SSBBCase.r_squared(linearYPred))}")
print(f"Determination coefficient = {SSBBCase.r_squared(linearYPred)}\n")
print("Predictions of known values")
unknownPred = SSBBCase.predictions(linearParams, predictionValues)
for i in range(len(unknownPred)):
    print(f"Batch Size = {SSBBCase.predictorX[i]}, Machine Efficiency Predicted = {linearYPred[i]}, Machine Efficiency = {SSBBCase.responseY[i]}")
print("\nPredictions of unknown values")
for i in range(len(unknownPred)):
    print(f"Batch Size = {predictionValues[i]}, Machine Efficiency = {unknownPred[i]}")

# Quadratic Regression
XMatrix = SSBBCase.initXMatrix(2)
BMatrix = SSBBCase.BMatrix(XMatrix, YMatrix)
quadraticParams = BMatrix
quadraticYPred = SSBBCase.predictions(quadraticParams,SSBBCase.predictorX)
print("\n\nQuadratic regression equation")
print(f"Machine Efficiency  = {float(quadraticParams[0])} + {float(quadraticParams[1])} Batch Size + {float(quadraticParams[2])} Batch Size^2")
print(f"Correlation coefficient = {SSBBCase.correlation_coefficient(SSBBCase.r_squared(quadraticYPred))}")
print(f"Determination coefficient = {SSBBCase.r_squared(quadraticYPred)}\n")
unknownPred = SSBBCase.predictions(quadraticParams, predictionValues)
print("Predictions of known values")
for i in range(len(unknownPred)):
    print(f"Batch Size = {SSBBCase.predictorX[i]}, Machine Efficiency Predicted = {quadraticYPred[i]}, Machine Efficiency = {SSBBCase.responseY[i]}")
print("\nPredictions of unknown values")
for i in range(len(unknownPred)):
    print(f"Batch Size = {predictionValues[i]}, Machine Efficiency = {unknownPred[i]}")

# Cubic Regression
XMatrix = SSBBCase.initXMatrix(3)
BMatrix = SSBBCase.BMatrix(XMatrix, YMatrix)
cubicParams = BMatrix
cubicYPred = SSBBCase.predictions(cubicParams, SSBBCase.predictorX)
print("\n\nCubic regression equation")
print(f"Machine Efficiency  = {float(cubicParams[0])} + {float(cubicParams[1])} Batch Size + {float(cubicParams[2])} Batch Size^2 + {float(cubicParams[3])} Batch Size^3")
print(f"Correlation coefficient = {SSBBCase.correlation_coefficient(SSBBCase.r_squared(cubicYPred))}")
print(f"Determination coefficient = {SSBBCase.r_squared(cubicYPred)}\n")
unknownPred = SSBBCase.predictions(cubicParams, predictionValues)
print("Predictions of known values")
for i in range(len(unknownPred)):
    print(f"Batch Size = {SSBBCase.predictorX[i]}, Machine Efficiency Predicted = {cubicYPred[i]}, Machine Efficiency = {SSBBCase.responseY[i]}")
print("\nPredictions of unknown values")
for i in range(len(unknownPred)):
    print(f"Batch Size = {predictionValues[i]}, Machine Efficiency = {unknownPred[i]}")
