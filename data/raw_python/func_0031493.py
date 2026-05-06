def findCodons(seq, codons):
    """
    Find all instances of the codons in 'codons' in the given sequence.

    seq: A Bio.Seq.Seq instance.
    codons: A set of codon strings.

    Return: a generator yielding matching codon offsets.
    """
    seqLen = len(seq)
    start = 0
    while start < seqLen:
        triplet = str(seq[start:start + 3])
        if triplet in codons:
            yield start
        start = start + 3