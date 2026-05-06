def brkl2d(arr,interval):
    '''
        arr = ["color1","r1","g1","b1","a1","color2","r2","g2","b2","a2"]
        >>> brkl2d(arr,5)
        [{'color1': ['r1', 'g1', 'b1', 'a1']}, {'color2': ['r2', 'g2', 'b2', 'a2']}]
    '''
    lngth = arr.__len__()
    brkseqs = elel.init_range(0,lngth,interval)
    l = elel.broken_seqs(arr,brkseqs)
    d = elel.mapv(l,lambda ele:{ele[0]:ele[1:]})
    return(d)