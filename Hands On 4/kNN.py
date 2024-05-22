import math

class kNN():
    def __init__(self, x, y, nClass, k):
        self.x = x
        self.y = y
        self.nClass = nClass
        self.k = k
        if len(self.x) != len(self.y) and len(self.x) != len(self.nClass):
            raise ValueError("The lenghts in your dataset are incorrect")

    def mean(self, variable):
        return sum(variable)/len(variable)

    def standardDeviation(self, variable):
        mean = self.mean(variable)
        sumSquaredDifferences = sum((value - mean) ** 2 for value in variable)
        variance = sumSquaredDifferences / len(variable)
        return math.sqrt(variance)
        
    def standarization(self, variable, newNeighbor):
        standarizedData = []
        mean = self.mean(variable)
        std = self.standardDeviation(variable)
        for i in variable:
            standarizedData.append((i - mean)/std)
        newNeighbor = (newNeighbor - mean)/std
        return standarizedData, newNeighbor
    
    def findDistance(self, newNeighborX, newNeighborY):
        X, newNeighborX = self.standarization(self.x, newNeighborX)
        Y, newNeighborY = self.standarization(self.y, newNeighborY)
        print("X    Y    Distance")
        print("------------------")
        distance = []
        for i in range(len(X)):
            dist = math.sqrt(((newNeighborX - X[i])**2) + ((newNeighborY - Y[i])**2))
            distance.append(dist)
            print("{:.2f}   {:.2f}   {:.2f}".format(X[i], Y[i], dist))
        return {distance[i]: self.nClass[i] for i in range(len(distance))}
        
    def findNearestNeighbors(self, newNeighborX, newNeighborY):
        distances = self.findDistance(newNeighborX, newNeighborY)
        sorted_distances = sorted(distances.items())
        k_nearest = sorted_distances[:self.k]
        # Obtener las clases de los k vecinos más cercanos
        nearest_classes = [neighbor[1] for neighbor in k_nearest]
    
        # Contar las ocurrencias de cada clase
        class_counts = {}
        for cls in nearest_classes:
            if cls in class_counts:
                class_counts[cls] += 1
            else:
                class_counts[cls] = 1
        
        # Encontrar la clase que se repite más veces
        max_class = max(class_counts, key=class_counts.get)
        return max_class

heights = [158, 158, 158, 160, 160, 
           163, 163, 160, 163, 165, 
           165, 165, 168, 168, 168, 
            170, 170, 170]

weights = [58, 59, 63, 59, 60,
           60, 61, 64, 64, 61, 
           62, 65, 62, 63, 66,
           63, 64, 68]

tShirtSize = ["M", "M", "M", "M", "M",
              "M", "M", "L", "L", "L", 
              "L", "L", "L", "L", "L", 
              "L", "L", "L"]

newNeighbor = [171, 65]
knn = kNN(heights, weights,tShirtSize,5 )
print(knn.findNearestNeighbors(newNeighbor[0], newNeighbor[1]))