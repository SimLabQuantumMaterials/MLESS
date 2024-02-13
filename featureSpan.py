import numpy as np

def computeMean(HEList, featureList):
    X = []
    Y = []

    for i in range(len(HEList)):
        Y.append(HEList[i][3])

        x = HEList[i][2]
        X1 = featureList[int(HEList[i][0]-57)]
        X2 = featureList[int(HEList[i][1]-57)]

        # Weighted mean, Quadratically weighted mean, absolute mean
        X.append(np.array([x*X1 + (1-x)*X2, (x**2*X1 + (1-x)**2*X2)/(x**2+(1-x)**2), abs(X1-X2)/2]).flatten())

    return X, Y


def generateFeatures(HEList, featureList):
    X = []
    Y = []

    for i in range(len(HEList)):
        Y.append(HEList[i][3])

        x = HEList[i][2]
        X1 = featureList[int(HEList[i][0]-57)]
        X2 = featureList[int(HEList[i][1]-57)]

        X.append(np.concatenate((np.array([x, 1/x, x * x, 1 / (x * x), 1-x, 1/(1-x), (1-x) * (1-x), 1/( (1-x) * (1-x))]), np.array([abs(X1-X2)/2.0, 2.0 / (abs(X1-X2)), (abs(X1-X2)/2.0) * (abs(X1-X2)/2.0), 4 / (abs(X1-X2) * abs(X1-X2)), (X1 + X2) / 2.0, 2.0 / (X1 + X2), (X1 + X2) * (X1 + X2) / 4, 4 / ((X1 + X2) * (X1 + X2))]).flatten())))
        
    return X, Y

def generateFeatures_2(HEList, featureList):
    X = []
    Y = []

    for i in range(len(HEList)):
        Y.append(HEList[i][3])

        x = HEList[i][2]
        X1 = featureList[int(HEList[i][0]-57)]
        X2 = featureList[int(HEList[i][1]-57)]

        X.append(np.concatenate((np.array([x*(1-x)]), np.array([abs(X1-X2)/2.0, 2.0 / (abs(X1-X2)), (abs(X1-X2)/2.0) * (abs(X1-X2)/2.0), 4 / (abs(X1-X2) * abs(X1-X2)), (X1 + X2) / 2.0, 2.0 / (X1 + X2), (X1 + X2) * (X1 + X2) / 4, 4 / ((X1 + X2) * (X1 + X2))]).flatten())))
        
    return X, Y

def X_Standardization(X):
    # contains the means for each feature
    mean = np.mean(X, axis = 0)
    # contains the standard deviation for each feathure
    std = np.std(X, axis=0)
    
    X_standardized=np.empty(shape=X.shape)
    #standardization
    for j in range(X.shape[1]):
        X_standardized[:,j] = (X[:,j] - np.ones(X.shape[0]) * mean[j]) / std[j]
        
    return X_standardized, mean, std
