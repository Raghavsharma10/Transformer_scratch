def calmarnorm(sharpe, T, tau = 1.0):
    '''
    Multiplicator for normalizing calmar ratio to period tau
    '''
    return calmar(sharpe,tau)/calmar(sharpe,T)