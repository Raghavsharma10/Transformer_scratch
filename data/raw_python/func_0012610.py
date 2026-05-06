def makeAngLenHist(ad, ld, fm = None, collapse=True, fit=spline_base.fit2d):
    """
    Histograms and performs a spline fit on the given data, 
    usually angle and length differences.
    
    Parameters:
        ad : array
            The data to be histogrammed along the x-axis. 
            May range from -180 to 180.
        ld : array
            The data to be histogrammed along the y-axis.
            May range from -36 to 36.
        collapse : boolean
            If true, the histogrammed data will include 
            negative values on the x-axis. Else, the histogram
            will be collapsed along x = 0, and thus contain only 
            positive angle differences
        fit : function or None, optional
            The function to use in order to fit the data. 
            If no fit should be applied, set to None
        fm  : fixmat or None, optional
            If given, the angle and length differences are calculated
            from the fixmat and the previous parameters are overwritten.
    """
    
    if fm:
        ad,ld = anglendiff(fm, roll=2)
        ad, ld = ad[0], ld[0]
        
    ld = ld[~np.isnan(ld)]
    ad = reshift(ad[~np.isnan(ad)])

    if collapse:
        e_y = np.linspace(-36.5, 36.5, 74)
        e_x = np.linspace(0, 180, 181)
        H = makeHist(abs(ad), ld, fit=fit, bins=[e_y, e_x])

        H = H/H.sum()
        
        return H
    else:
        e_x = np.linspace(-180, 180, 361)
        e_y = np.linspace(-36.5, 36.5, 74)
        return makeHist(ad, ld, fit=fit, bins=[e_y, e_x])