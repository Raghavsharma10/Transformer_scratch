def generate_sub_codons_left(codons_dict):
    """Generate the sub_codons_left dictionary of codon prefixes.

    Parameters
    ----------
    codons_dict : dict
        Dictionary, keyed by the allowed 'amino acid' symbols with the values 
        being lists of codons corresponding to the symbol.

    Returns
    -------
    sub_codons_left : dict
        Dictionary of the 1 and 2 nucleotide prefixes (read from 5') for 
        each codon in an 'amino acid' grouping
        
    """
    sub_codons_left = {}
    for aa in codons_dict.keys():
        sub_codons_left[aa] = list(set([x[0] for x in codons_dict[aa]] + [x[:2] for x in codons_dict[aa]]))
        
    return sub_codons_left