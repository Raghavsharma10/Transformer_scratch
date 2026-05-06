def calmar(sharpe, T = 1.0):
    '''
    Calculate the Calmar ratio for a Weiner process
    
    @param sharpe:    Annualized Sharpe ratio
    @param T:         Time interval in years
    '''
    x = 0.5*T*sharpe*sharpe
    return x/qp(x)