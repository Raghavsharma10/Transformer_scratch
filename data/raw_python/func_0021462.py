def cutR_seq(seq, cutR, max_palindrome):
    """Cut genomic sequence from the right.

    Parameters
    ----------
    seq : str
        Nucleotide sequence to be cut from the right
    cutR : int
        cutR - max_palindrome = how many nucleotides to cut from the right.
        Negative cutR implies complementary palindromic insertions.
    max_palindrome : int
        Length of the maximum palindromic insertion.

    Returns
    -------
    seq : str
        Nucleotide sequence after being cut from the right
    
    Examples
    --------
    >>> cutR_seq('TGCGCCAGCAGTGAGTC', 0, 4)
    'TGCGCCAGCAGTGAGTCGACT'
    >>> cutR_seq('TGCGCCAGCAGTGAGTC', 8, 4)
    'TGCGCCAGCAGTG'
    
    """
    complement_dict = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'} #can include lower case if wanted
    if cutR < max_palindrome:
        seq = seq + ''.join([complement_dict[nt] for nt in seq[cutR - max_palindrome:]][::-1]) #reverse complement palindrome insertions
    else:
        seq = seq[:len(seq) - cutR + max_palindrome] #deletions
    
    return seq