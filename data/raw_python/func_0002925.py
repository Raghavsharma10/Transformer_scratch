def calcMse(predTst, yTest, axis=0):
    """calculate mean squared error. Assumes that axis=0 is time

        Parameters
        ----------
        predTst : np.array, predicted reponse for yTest
        yTest : np.array, acxtually observed response for yTest
        Returns
        -------
        aryFunc : np.array
            MSE
    """
    return np.mean((yTest - predTst) ** 2, axis=axis)