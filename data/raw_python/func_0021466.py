def determine_seq_type(seq, aa_alphabet):
    """Determine the type of a sequence.
    
    Parameters
    ----------

    seq : str
        Sequence to be typed.
    aa_alphabet : str
        String of all characters recoginized as 'amino acids'. (i.e. the keys
        of codons_dict: aa_alphabet = ''.join(codons_dict.keys())  )

    Returns
    -------
    seq_type : str
        The type of sequence (ntseq, aaseq, regex, None) seq is.
    
    Example
    --------
    >>> determine_seq_type('TGTGCCAGCAGTTCCGAAGGGGCGGGAGGGCCCTCCCTGAGAGGTCATGAGCAGTTCTTC', aa_alphabet)
    'ntseq'
    >>> determine_seq_type('CSARDX[TV]GNX{0,}', aa_alphabet)
    'regex
    
    """
    
    if all([x in 'ACGTacgt' for x in seq]):
        return 'ntseq'
    elif all([x in aa_alphabet for x in seq]):
        return 'aaseq'
    elif all([x in aa_alphabet + '[]{}0123456789,']):
        return 'regex'