def nt2codon_rep(ntseq):
    """Represent nucleotide sequence by sequence of codon symbols.

    'Translates' the nucleotide sequence into a symbolic representation of
    'amino acids' where each codon gets its own unique character symbol. These 
    characters should be reserved only for representing the 64 individual 
    codons --- note that this means it is important that this function matches 
    the corresponding function in the preprocess script and that any custom 
    alphabet does not use these symbols. Defining symbols for each individual
    codon allows for Pgen computation of inframe nucleotide sequences. 

    Parameters
    ----------

    ntseq : str
        A Nucleotide sequence (normally a CDR3 nucleotide sequence) to be 
        'translated' into the codon - symbol representation. Can be either
        uppercase or lowercase, but only composed of A, C, G, or T.

    Returns
    -------
    codon_rep : str
        The codon - symbolic representation of ntseq. Note that if 
        len(ntseq) == 3L --> len(codon_rep) == L
    
    Example
    --------
    >>> nt2codon_rep('TGTGCCTGGAGTGTAGCTCCGGACAGGGGTGGCTACACCTTC')
    '\xbb\x96\xab\xb8\x8e\xb6\xa5\x92\xa8\xba\x9a\x93\x94\x9f'
        
    """
    
    #
    nt2num = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
    #Use single characters not in use to represent each individual codon --- this function is called in constructing the codon dictionary
    codon_rep ='\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f\x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf'
    
    return ''.join([codon_rep[nt2num[ntseq[i]] + 4*nt2num[ntseq[i+1]] + 16*nt2num[ntseq[i+2]]] for i in range(0, len(ntseq), 3) if i+2 < len(ntseq)])