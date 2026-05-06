def makeHist(x_val, y_val, fit=spline_base.fit2d, 
            bins=[np.linspace(-36.5,36.5,74),np.linspace(-180,180,361)]):
    """
    Constructs a (fitted) histogram of the given data.
    
    Parameters:
        x_val : array
            The data to be histogrammed along the x-axis. 
        y_val : array
            The data to be histogrammed along the y-axis.
        fit : function or None, optional
            The function to use in order to fit the data. 
            If no fit should be applied, set to None
        bins : touple of arrays, giving the bin edges to be 
            used in the histogram. (First value: y-axis, Second value: x-axis)
    """
    
    y_val = y_val[~np.isnan(y_val)]
    x_val = x_val[~np.isnan(x_val)]
    
    samples = list(zip(y_val, x_val))
    K, xedges, yedges = np.histogram2d(y_val, x_val, bins=bins)

    if (fit is None):
        return K/ K.sum()
   
    # Check if given attr is a function
    elif hasattr(fit, '__call__'):
        H = fit(np.array(samples), bins[0], bins[1], p_est=K)[0]
        return H/H.sum()
    else:
        raise TypeError("Not a valid argument, insert spline function or None")