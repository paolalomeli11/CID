from math import sqrt
class SimpleLinearRegression:
    def __init__(self, independent, dependent):
        # Check if the lengths of independent and dependent variables match
        if len(independent) != len(dependent):
            raise ValueError("The lengths of independent and dependent do not match.")
        self.independent = independent
        self.dependent = dependent

    def sumIndependentDependent(self):
        return sum(x * y for x,y in zip(self.independent, self.dependent))

    def sumIndependentSquared(self):
        return sum(x ** 2 for x in self.independent)
    
    def B1(self):
        n = len(self.independent)
        return ((n * self.sumIndependentDependent()) - (sum(self.independent) * sum(self.dependent))) / ((n * self.sumIndependentSquared()) - (sum(self.independent) ** 2))
    
    def B0(self):
        return (sum(self.dependent) - (self.B1() * sum(self.independent))) / len(self.independent)
    
    def mean(self, variable):
        return sum(variable)/len(variable)

    def variance(self, variable):
        return sum((x - self.mean(variable)) ** 2 for x in variable)/len(variable)

    def covariance(self):
        return sum((x - self.mean(self.independent)) * (y - self.mean(self.dependent)) for x,y in zip(self.independent, self.dependent))/len(self.independent)
    
    def correlationCoefficient(self):
        return (self.covariance()) / sqrt(self.variance(self.independent) * self.variance(self.dependent))
    
    def determinationCoefficient(self):
        return self.covariance() **2 /(self.variance(self.dependent) * self.variance(self.independent))
    
    def SSR(self, variable):
        return sum((x - self.mean(self.dependent)) ** 2 for x in variable)

    def SST(self):
        return self.variance(self.dependent)*len(self.dependent)
    
    def SSE(self, variable):
        return sum((y - y1) ** 2 for y,y1 in zip(self.dependent, variable))

# Linear Regression Case Benetton
# Dependent data
sales = (6,12,18,24,30,36,42,48,54)
# Independent data
advertising = (1,2,3,4,5,6,7,8,9)

# Prediction values
predictionValues = (0, 15, 20, 60, 80)

# Creating instance of SimpleLinearRegression class
caseBenetton = SimpleLinearRegression(advertising, sales)

# Calculating coefficients of the linear regression equation
b0, b1 = caseBenetton.B0(), caseBenetton.B1()

# Printing coefficients
print(f"B0 = {b0}")
print(f"B1 = {b1}")

# Printing the linear regression equation
print(f"y = {b0} + {b1}x")

# Calculating and printing the correlation coefficient
print(f"Correlation coefficient = {caseBenetton.correlationCoefficient()}")

# Calculating and printing the determination coefficient
print(f"Determination coefficient = {caseBenetton.determinationCoefficient()}\n")

# Predicting sales for each prediction value
for i in predictionValues:
    y = b0 + b1 * i
    print(f"Input value X = {i}")
    print(f"Predicted value Y = {y}\n")

salesPredicted = []
for i in range(len(advertising)):
    salesPredicted.append(b0 + b1 * advertising[i])

print(f"SST = {caseBenetton.SST()}")
print(f"SSR = {caseBenetton.SSR(salesPredicted)}")
print(f"SSE = {caseBenetton.SSE(salesPredicted)}")

