def simulateAR1(n,
                beta,
                sigma,
                c,
                burnin,
                varNumCNR,
                varNumTP,
                ):
    """
    Simulates an AR(1) model using the parameters beta, c, and sigma.
    Returns an array with length n

    n := number of time points
    beta := correlation weight
    sigma := standard deviation of the noise, can be a vector
    c := constant added to the noise, default 0

    based on:
    source1: https://github.com/ndronen/misc/blob/master/python/ar1.py
    source2: http://stats.stackexchange.com/questions/22742/
             problem-simulating-ar2-process
    source3: https://kurtverstegen.wordpress.com/2013/12/07/simulation/
    """
    # Output array with noise time courses
    noise = np.empty((varNumCNR, varNumTP))
    if burnin == 1:
        burnin = 100
        n = n + burnin

    noiseTemp = c + sp.random.normal(0, 1, n)
    sims = np.zeros(n)
    sims[0] = noiseTemp[0]
    for i in range(1, n):
        sims[i] = beta*sims[i-1] + noiseTemp[i]
    sims = sims[burnin:]
    noise = sigma[:, np.newaxis]*sp.stats.mstats.zscore(sims)
    return noise