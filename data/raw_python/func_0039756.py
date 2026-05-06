def sigmafilter(data, sigmas, passes):
    """Remove datapoints outside of a specified standard deviation range."""
    for n in range(passes):
        meandata = np.mean(data[~np.isnan(data)])
        sigma = np.std(data[~np.isnan(data)])
        data[data > meandata+sigmas*sigma] = np.nan
        data[data < meandata-sigmas*sigma] = np.nan
    return data