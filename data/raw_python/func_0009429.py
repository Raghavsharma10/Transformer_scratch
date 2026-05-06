def function(x, ax, ay):
    '''
    general square root function
    '''
    with np.errstate(invalid='ignore'):
        return ay * (x - ax)**0.5