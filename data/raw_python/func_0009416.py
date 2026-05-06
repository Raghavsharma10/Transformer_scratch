def exampleSignals(std=1, dur1=1, dur2=3, dur3=0.2,
                          n1=0.5, n2=0.5, n3=2):
    '''
    std ... standard deviation of every signal
    dur1...dur3 --> event duration per second
    n1...n3 --> number of events per second
    '''
    np.random.seed(123)
    t = np.linspace(0, 10, 100)

    f0 = _flux(t, n1, dur1, std, offs=0)
    f1 = _flux(t, n2, dur2, std, offs=0)
    f2 = _flux(t, n3, dur3, std, offs=0)
    return t,f0,f1,f2