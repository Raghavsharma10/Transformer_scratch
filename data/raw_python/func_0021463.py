def cutL_seq(seq, cutL, max_palindrome):
    """Cut genomic sequence from the left.

    Parameters
    ----------
    seq : str
        Nucleotide sequence to be cut from the right
    cutL : int
        cutL - max_palindrome = how many nucleotides to cut from the left.
        Negative cutL implies complementary palindromic insertions.
    max_palindrome : int
        Length of the maximum palindromic insertion.

    Returns
    -------
    seq : str
        Nucleotide sequence after being cut from the left
    
    Examples
    --------
    >>> cutL_seq('TGAACACTGAAGCTTTCTTT', 8, 4)
    'CACTGAAGCTTTCTTT'
    >>> cutL_seq('TGAACACTGAAGCTTTCTTT', 0, 4)
    'TTCATGAACACTGAAGCTTTCTTT'
    
    """
    
    complement_dict = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'} #can include lower case if wanted
    if cutL < max_palindrome:
        seq = ''.join([complement_dict[nt] for nt in seq[:max_palindrome - cutL]][::-1]) + seq #reverse complement palindrome insertions
    else:
        seq = seq[cutL-max_palindrome:] #deletions
    
    return seq