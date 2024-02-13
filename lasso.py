from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import numpy as np
import Utils
import featureSpan
from itertools import combinations, product


##### Define a function for the fitting of Lasso with a given $\lambda$
def LassoFit(lmb, X, Y, max_iter=100000, standardization = True):
    if standardization:
        X_standardized, mean, std = featureSpan.X_Standardization(X) 
        #print("X is standardized")
    else:
        X_standardized = X
    lasso =  Lasso(alpha=lmb, max_iter=max_iter)
    lasso.fit(X_standardized.copy(), Y.copy())
    coef =  lasso.coef_
    selected_indices = coef.nonzero()[0]
    Y_predict = lasso.predict(X_standardized)
    MAE, MSE, ME, _1, _2 = Utils.compute_error(Y.copy(), Y_predict)
    
    return coef, selected_indices, MAE, MSE, ME


def LassoPlot(X, Y, min, max, step, standardization = True ):
    if standardization:
        X_standardized, mean, std = featureSpan.X_Standardization(X) 
        #print("X is standardized")
    else:
        X_standardized = X
    coefs = []
    indices = []
    MAEs = []
    MSEs = []
    MEs = []
    lmbs = []
    nbs = []

    for lmbda in np.arange (min, max, step):
        lmbs.append(np.array(lmbda))
        coef, selected_indices, MAE, MSE, ME = LassoFit(lmbda, X_standardized, Y)
        coefs.append(np.array(coef))
        indices.append(np.array(selected_indices))
        nbs.append(len(selected_indices))
        MAEs.append(np.array(MAE))
        MSEs.append(np.array(MSE))
        MEs.append(np.array(ME))
    
    plt.figure(figsize=(24, 6))
    plt.subplot(131)
    plt.plot(lmbs,MAEs,label="MAE", marker="o")
    plt.plot(lmbs,MSEs, label="MSE", marker="o")
    plt.xlabel('Lambda')
    plt.ylabel('Error')
    plt.legend()
    plt.subplot(132)
    plt.plot(lmbs,MEs, label="ME",marker="o")
    plt.xlabel('Lambda')
    plt.ylabel('Error')
    plt.legend()
    plt.subplot(133)
    plt.plot(lmbs,nbs, label="number of descriptors",marker="o")
    plt.xlabel('Lambda')
    plt.ylabel('number of descriptors')
    plt.legend()
    
    
#### Define a function which fits Lasso to have no more nonzero coefficients than a given threshold 
def LassoSelect(X, Y, feature_list, min, max, step, threshold, standardization = True):
    if standardization:
        X_standardized, mean, std = featureSpan.X_Standardization(X) 
        #print("X is standardized")
    else:
        X_standardized = X
        
    found = False
    for lmbda in np.arange (min, max, step):
        coef, selected_indices, MAE, MSE, ME = LassoFit(lmbda, X.copy(), Y.copy())
        if len(selected_indices) <= threshold:
            found = True
            break
    
    if found:
        print("FOUND with threshold: {}".format(threshold))
        print("Lambda: {}, nnz: {}, MAE: {}, MSE: {}, ME: {}".format(lmbda, len(selected_indices), MAE, MSE, ME))
            
    else:
        print("NOT FOUND with threshold: {}".format(threshold))
        print("Closest are: ")
        print("Lambda: {}, nnz: {}, MAE: {}, MSE: {}, ME: {}".format(lmbda, len(selected_indices), MAE, MSE, ME))
     
    X_reduced = X[:,selected_indices]
    if standardization:
        mean_reduced = mean[selected_indices]
        std_reduced = std[selected_indices]
    
    feature_reduced = np.array(feature_list)[selected_indices]

    if standardization:
        return selected_indices, X_reduced, mean_reduced, std_reduced, feature_reduced
    else:
        return selected_indices, X_reduced, feature_reduced

### Lasso-l0
def LassoL0(X, Y, nnz):    
    nr, nc = X.shape
    X = np.column_stack((X, np.ones(nr)))
    se_min = np.inner(Y, Y)
    coef_min, permu_min = None, None
    for permu in combinations(range(nc), nnz):
        X_ls = X[:, permu + (-1,)]
        coef, se, __1, __2 = np.linalg.lstsq(X_ls, Y, rcond=-1)
        try:
            if se[0] < se_min: 
                se_min = se[0]
                coef_min, permu_min = coef, permu
        except:
            pass
        
    return coef_min, permu_min

###

def LassoL0Fit(X_reduced, Y, nnz, mean_reduced, std_reduced, feature_reduced, log=True):
    nr, nc = X_reduced.shape
   
    X_std = np.empty(shape = X_reduced.shape)
    
    for j in range(X_reduced.shape[1]):
        X_std[:,j] = (X_reduced[:,j] - np.ones(X_reduced.shape[0]) * mean_reduced[j]) / std_reduced[j]
        
    coefficients, selected_indices = LassoL0(X_std, Y, nnz)
    
    coefficients = np.array(coefficients)
    selected_indices = np.array(selected_indices)
    
    feature_list_selected = np.array(feature_reduced)[selected_indices]
    X_selected = X_reduced[:, selected_indices]
    mean_selected = mean_reduced[selected_indices]
    std_selected = std_reduced[selected_indices]
    
    if log:
        print("Lasso: selected coefficients are: {}".format(coefficients))
        print("Lasso: selected features are: {}".format(feature_list_selected))
        
    #-mean/std
    mean_std = []
    for i in range(len(selected_indices)):
        mean_std.append(coefficients[i] * mean_selected[i]/std_selected[i])
 
    sum_mean_std = sum(mean_std)

    for i in range(len(selected_indices)):
        coefficients[i] = coefficients[i] / std_selected[i]

    
    coefficients[len(selected_indices)] -= sum_mean_std
    
    function = str(coefficients[0])+" * "+feature_list_selected[0]
    
    for i in range(1, len(selected_indices)):
        if coefficients[i] >= 0:
            function += " + " + str(coefficients[i])+" * "+feature_list_selected[i]
        else:
            function += " - " + str(abs(coefficients[i]))+" * "+feature_list_selected[i]

    
    if coefficients[len(selected_indices)] >= 0:
        function += " + " + str(coefficients[len(selected_indices)])
    else:
        function += " - " + str(abs(coefficients[len(selected_indices)]))
    
    if log:
        print("Constructed function is: {}".format(function))

    X_selected = np.column_stack((X_selected, np.ones(X_selected.shape[0])))
    Y_predict = X_selected[:,0] * coefficients[0]

    for i in range(1,len(selected_indices)+1):
        Y_predict = Y_predict + X_selected[:,i] * coefficients[i]
    
    if log:
        Utils.print_error(Y.copy(),Y_predict,"Lasso L0: {} coef".format(nnz))
    
    return Y_predict, coefficients, selected_indices

###
def LassoL0Fit_2(X_reduced, Y, nnz, feature_reduced, log=True):
    nr, nc = X_reduced.shape
        
    coefficients, selected_indices = LassoL0(X_reduced, Y, nnz)
    
    coefficients = np.array(coefficients)
    selected_indices = np.array(selected_indices)
    
    feature_list_selected = np.array(feature_reduced)[selected_indices]
    X_selected = X_reduced[:, selected_indices]
    
    if log:
        print("Lasso: selected coefficients are: {}".format(coefficients))
        print("Lasso: selected features are: {}".format(feature_list_selected))
    
    function = str(coefficients[0])+" * "+feature_list_selected[0]
    
    for i in range(1, len(selected_indices)):
        if coefficients[i] >= 0:
            function += " + " + str(coefficients[i])+" * "+feature_list_selected[i]
        else:
            function += " - " + str(abs(coefficients[i]))+" * "+feature_list_selected[i]

    
    if coefficients[len(selected_indices)] >= 0:
        function += " + " + str(coefficients[len(selected_indices)])
    else:
        function += " - " + str(abs(coefficients[len(selected_indices)]))
    
    if log:
        print("Constructed function is: {}".format(function))

    X_selected = np.column_stack((X_selected, np.ones(X_selected.shape[0])))
    Y_predict = X_selected[:,0] * coefficients[0]

    for i in range(1,len(selected_indices)+1):
        Y_predict = Y_predict + X_selected[:,i] * coefficients[i]
    
    if log:
        Utils.print_error(Y.copy(),Y_predict,"Lasso L0: {} coef".format(nnz))
    
    return Y_predict, coefficients, selected_indices