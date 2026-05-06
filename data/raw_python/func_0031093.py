def aa3_to_aa1(seq):
    """convert string of 3-letter amino acids to 1-letter amino acids

    >>> aa3_to_aa1("CysAlaThrSerAlaArgGluLeuAlaMetGlu")
    'CATSARELAME'

    >>> aa3_to_aa1(None)

    """
    if seq is None:
        return None
    return "".join(aa3_to_aa1_lut[aa3]
                   for aa3 in [seq[i:i + 3] for i in range(0, len(seq), 3)])