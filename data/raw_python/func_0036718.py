def __read_single_fasta_query_lines(f):
    '''
    Read and return sequence of lines (including newlines) that
    represent a single FASTA query record. The provided file is
    expected to be blocking.

    Returns None if there are no more query sequences in the file.
    '''

    def readline():
        l = f.readline()
        if l == '':
            raise EOFError()
        return l

    rec = None
    try:
        l = readline()
        assert l.startswith('>')

        rec = [l]
        while True:
            pos = f.tell()
            l = readline()
            if l.startswith('>'):
                f.seek(pos, 0)
                break
            rec += [l]
    except EOFError:
        pass

    return rec