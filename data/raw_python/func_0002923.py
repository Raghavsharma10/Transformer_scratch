def calcR2(predTst, yTest, axis=0):
    """calculate coefficient of determination. Assumes that axis=0 is time

        Parameters
        ----------
        predTst : np.array, predicted reponse for yTest
        yTest : np.array, acxtually observed response for yTest
        Returns
        -------
        aryFunc : np.array
            R2
    """
    rss = np.sum((yTest - predTst) ** 2, axis=axis)
    tss = np.sum((yTest - yTest.mean()) ** 2, axis=axis)

    return 1 - rss/tss