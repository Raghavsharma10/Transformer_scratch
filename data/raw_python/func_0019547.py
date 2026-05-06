def motif_from_consensus(cons, n=12):
    """Convert consensus sequence to motif.

    Converts a consensus sequences using the nucleotide IUPAC alphabet to a 
    motif. 

    Parameters
    ----------
    cons : str
        Consensus sequence using the IUPAC alphabet.
    n : int , optional
        Count used to convert the sequence to a PFM.
    
    Returns
    -------
    m : Motif instance
        Motif created from the consensus.
    """
    width = len(cons)
    nucs = {"A":0,"C":1,"G":2,"T":3}
    pfm = [[0 for _ in range(4)] for _ in range(width)]
    m = Motif()
    for i,char in enumerate(cons):
        for nuc in m.iupac[char.upper()]:
            pfm[i][nucs[nuc]] = n / len(m.iupac[char.upper()])
    m = Motif(pfm)
    m.id = cons
    return m