def motif_from_align(align):
    """Convert alignment to motif.

    Converts a list with sequences to a motif. Sequences should be the same 
    length.

    Parameters
    ----------
    align : list
        List with sequences (A,C,G,T).
    
    Returns
    -------
    m : Motif instance
        Motif created from the aligned sequences.
    """
    width = len(align[0])
    nucs = {"A":0,"C":1,"G":2,"T":3}
    pfm =  [[0 for _ in range(4)] for _ in range(width)]
    for row in align:
        for i in range(len(row)):
            pfm[i][nucs[row[i]]] += 1
    m = Motif(pfm)
    m.align = align[:]
    return m