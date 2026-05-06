def needle(reads):
    """
    Run a Needleman-Wunsch alignment and return the two sequences.

    @param reads: An iterable of two reads.
    @return: A C{Reads} instance with the two aligned sequences.
    """
    from tempfile import mkdtemp
    from shutil import rmtree

    dir = mkdtemp()

    file1 = join(dir, 'file1.fasta')
    with open(file1, 'w') as fp:
        print(reads[0].toString('fasta'), end='', file=fp)

    file2 = join(dir, 'file2.fasta')
    with open(file2, 'w') as fp:
        print(reads[1].toString('fasta'), end='', file=fp)

    out = join(dir, 'result.fasta')

    Executor().execute("needle -asequence '%s' -bsequence '%s' -auto "
                       "-outfile '%s' -aformat fasta" % (
                           file1, file2, out))

    # Use 'list' in the following to force reading the FASTA from disk.
    result = Reads(list(FastaReads(out)))
    rmtree(dir)

    return result