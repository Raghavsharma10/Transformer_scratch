def firstSacDist(fm):
    """
        Computes the distribution of angle and length
        combinations that were made as first saccades
        
        Parameters:
            fm : ocupy.fixmat 
                The fixation data to be analysed
    
    """  
    ang, leng, ad, ld = anglendiff(fm, return_abs=True)
    y_arg = leng[0][np.roll(fm.fix == min(fm.fix), 1)]/fm.pixels_per_degree
    x_arg = reshift(ang[0][np.roll(fm.fix == min(fm.fix), 1)])
    bins = [list(range(int(ceil(np.nanmax(y_arg)))+1)), np.linspace(-180, 180, 361)]
    return makeHist(x_arg, y_arg, fit=None, bins = bins)