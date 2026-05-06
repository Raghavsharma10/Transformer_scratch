def calcFstats(predTst, yTest, p, axis=0):
    """calculate coefficient of determination. Assumes that axis=0 is time

        Parameters
        ----------
        predTst : np.array, predicted reponse for yTest
        yTest : np.array, acxtually observed response for yTest
        p: float, number of predictors
        Returns
        -------
        aryFunc : np.array
            R2
    """
    rss = np.sum((yTest - predTst) ** 2, axis=axis)
    tss = np.sum((yTest - yTest.mean()) ** 2, axis=axis)
    # derive number of measurements
    n = yTest.shape[0]
    # calculate Fvalues
    vecFvals = ((tss - rss)/p)/(rss/(n-p-1))
    # calculate corresponding po values
    df1 = p - 1
    df2 = n-1
    vecPvals = stats.f.cdf(vecFvals, df1, df2)

    return vecFvals, vecPvals