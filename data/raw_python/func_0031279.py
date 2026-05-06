def findOrDie(s):
    """
    Look up an amino acid.

    @param s: A C{str} amino acid specifier. This may be a full name,
        a 3-letter abbreviation or a 1-letter abbreviation. Case is ignored.
    @return: An C{AminoAcid} instance, if one can be found. Else exit.
    """
    aa = find(s)
    if aa:
        return aa
    else:
        print('Unknown amino acid or codon: %s' % s, file=sys.stderr)
        print('Valid arguments are: %s.' % list(CODONS.keys()),
              file=sys.stderr)
        sys.exit(1)