def trajLenDist(fm):
    """
        Computes the distribution of trajectory lengths, i.e.
        the number of saccades that were made as a part of one trajectory
        
        Parameters:
            fm : ocupy.fixmat 
                The fixation data to be analysed
    
    """  
    trajLen = np.roll(fm.fix, 1)[fm.fix == min(fm.fix)]
    val, borders = np.histogram(trajLen, 
                    bins=np.linspace(-0.5, max(trajLen)+0.5, max(trajLen)+2))
    cumsum = np.cumsum(val.astype(float) / val.sum())
    return cumsum, borders