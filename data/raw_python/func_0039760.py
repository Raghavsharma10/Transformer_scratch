def calcstats(data, t1, t2, sr):
    """Calculate the mean and standard deviation of some array between
    t1 and t2 provided the sample rate sr.
    """
    dataseg = data[sr*t1:sr*t2]
    meandata = np.mean(dataseg[~np.isnan(dataseg)])
    stddata = np.std(dataseg[~np.isnan(dataseg)])
    return meandata, stddata