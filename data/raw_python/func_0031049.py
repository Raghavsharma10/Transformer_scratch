def dedupFasta(reads):
    """
    Remove sequence duplicates (based on sequence) from FASTA.

    @param reads: a C{dark.reads.Reads} instance.
    @return: a generator of C{dark.reads.Read} instances with no duplicates.
    """
    seen = set()
    add = seen.add
    for read in reads:
        hash_ = md5(read.sequence.encode('UTF-8')).digest()
        if hash_ not in seen:
            add(hash_)
            yield read