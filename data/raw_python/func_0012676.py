def nss(prediction, fix):
    """
    Compute the normalized scanpath salience

    input:
        fix : list, l[0] contains y, l[1] contains x
    """

    prediction = prediction - np.mean(prediction)
    prediction = prediction / np.std(prediction)
    return np.mean(prediction[fix[0], fix[1]])