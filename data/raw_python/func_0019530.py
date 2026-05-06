def ssd(p1, p2):
    """Calculates motif position similarity based on sum of squared distances.
    
    Parameters
    ----------
    p1 : list
        Motif position 1.
    
    p2 : list
        Motif position 2.
    
    Returns
    -------
    score : float
    """
    return 2 - np.sum([(a-b)**2 for a,b in zip(p1,p2)])