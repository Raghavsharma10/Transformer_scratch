def nt2aa(ntseq):
    """Translate a nucleotide sequence into an amino acid sequence.

    Parameters
    ----------
    ntseq : str
        Nucleotide sequence composed of A, C, G, or T (uppercase or lowercase)

    Returns
    -------
    aaseq : str
        Amino acid sequence
    
    Example
    --------
    >>> nt2aa('TGTGCCTGGAGTGTAGCTCCGGACAGGGGTGGCTACACCTTC')
    'CAWSVAPDRGGYTF'
        
    """
    nt2num = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
    aa_dict ='KQE*TPASRRG*ILVLNHDYTPASSRGCILVFKQE*TPASRRGWMLVLNHDYTPASSRGCILVF'
    
    return ''.join([aa_dict[nt2num[ntseq[i]] + 4*nt2num[ntseq[i+1]] + 16*nt2num[ntseq[i+2]]] for i in range(0, len(ntseq), 3) if i+2 < len(ntseq)])