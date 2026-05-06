def translate_cds(seq, full_codons=True, ter_symbol="*"):
    """translate a DNA or RNA sequence into a single-letter amino acid sequence
    using the standard translation table

    If full_codons is True, a sequence whose length isn't a multiple of three
    generates a ValueError; else an 'X' will be added as the last amino acid.
    This matches biopython's behaviour when padding the last codon with 'N's.

    >>> translate_cds("ATGCGA")
    'MR'

    >>> translate_cds("AUGCGA")
    'MR'

    >>> translate_cds(None)


    >>> translate_cds("")
    ''

    >>> translate_cds("AUGCG")
    Traceback (most recent call last):
    ...
    ValueError: Sequence length must be a multiple of three

    >>> translate_cds("AUGCG", full_codons=False)
    'M*'

    >>> translate_cds("AUGCGQ")
    Traceback (most recent call last):
    ...
    ValueError: Codon CGQ at position 4..6 is undefined in codon table

    """
    if seq is None:
        return None

    if len(seq) == 0:
        return ""

    if full_codons and len(seq) % 3 != 0:
        raise ValueError("Sequence length must be a multiple of three")

    seq = replace_u_to_t(seq)
    seq = seq.upper()

    protein_seq = list()
    for i in range(0, len(seq) - len(seq) % 3, 3):
        try:
            aa = dna_to_aa1_lut[seq[i:i + 3]]
        except KeyError:
            raise ValueError("Codon {} at position {}..{} is undefined in codon table".format(
                seq[i:i + 3], i+1, i+3))
        protein_seq.append(aa)

    # check for trailing bases and add the ter symbol if required
    if not full_codons and len(seq) % 3 != 0:
        protein_seq.append(ter_symbol)

    return ''.join(protein_seq)