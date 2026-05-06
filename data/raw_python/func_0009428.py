def boundedFunction(x, minY, ax, ay):
    '''
    limit [function] to a minimum y value 
    '''
    y = function(x, ax, ay)
    return np.maximum(np.nan_to_num(y), minY)