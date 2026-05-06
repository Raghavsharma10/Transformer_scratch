def normalize_sequence(seq):
    """return normalized representation of sequence for hashing

    This really means ensuring that the sequence is represented as a
    binary blob and removing whitespace and asterisks and uppercasing.

    >>> normalize_sequence("ACGT")
    'ACGT'

    >>> normalize_sequence("  A C G T * ")
    'ACGT'

    >>> normalize_sequence("ACGT1")
    Traceback (most recent call last):
    ...
    RuntimeError: Normalized sequence contains non-alphabetic characters

    """

    nseq = re.sub(r"[\s\*]", "", seq).upper()
    m = re.search("[^A-Z]", nseq)
    if m:
        _logger.debug("Original sequence: " + seq)
        _logger.debug("Normalized sequence: " + nseq)
        _logger.debug("First non-[A-Z] at {}".format(m.start()))
        raise RuntimeError("Normalized sequence contains non-alphabetic characters")
    return nseq