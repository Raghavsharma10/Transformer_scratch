def getPathogenProteinCounts(filenames):
    """
    Get the number of proteins for each pathogen in C{filenames}.

    @param filenames: Either C{None} or a C{list} of C{str} FASTA file names.
        If C{None} an empty C{Counter} is returned. If FASTA file names are
        given, their sequence ids should have the format used in the NCBI
        bacterial and viral protein reference sequence files, in which the
        protein name is followed by the pathogen name in square brackets.
    @return: A C{Counter} keyed by C{str} pathogen name, whose values are
        C{int}s with the count of the number of proteins for the pathogen.
    """
    result = Counter()
    if filenames:
        for filename in filenames:
            for protein in FastaReads(filename):
                _, pathogenName = splitNames(protein.id)
                if pathogenName != _NO_PATHOGEN_NAME:
                    result[pathogenName] += 1

    return result