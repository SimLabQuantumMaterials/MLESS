import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def mean_absolute_percentage_error_2(a, b):
    n = len(a)
    error = 0
    for i in range(n):
        error = error + abs(a[i] - b[i]) / abs(a[i])
        
    error = error / n
    
    return error


def print_error(a, b, Set):
    print(Set)
    print("=========")
    print("Mean absolute error: {}".format(
        mean_absolute_error(a, b)
    ))
    print("Mean squared error: {}".format(
        mean_squared_error(a, b)
    ))
    
    print("Mean absolute percentage error: {}".format(
        mean_absolute_percentage_error_2(a, b)
    ))

    
    
def compute_error(a, b):
    error = np.abs(a-b)
    max_error = np.argmax(error)
    return mean_absolute_error(a, b), mean_squared_error(a, b), mean_absolute_percentage_error_2(a, b)

#import numpy as np
#from sklearn.metrics import mean_absolute_error, mean_squared_error

#def print_error(a, b, Set):
#    print(Set)
#    print("=========")
#    print("Mean absolute error: {}".format(
#        mean_absolute_error(a, b)
#    ))
#    print("Mean squared error: {}".format(
#        mean_squared_error(a, b)
#    ))
#    error = np.abs(a-b)
#    max_error = np.argmax(error)

#    print("Max error: {}".format(error[max_error]))
#    print("True value: {}".format(a[max_error]))
#    print("Predicted value: {}".format(b[max_error]))
    
    
#def compute_error(a, b):
#    error = np.abs(a-b)
#    max_error = np.argmax(error)
#    return mean_absolute_error(a, b), mean_squared_error(a, b), error[max_error], a[max_error], b[max_error]