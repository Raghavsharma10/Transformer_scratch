def __read_single_query_result(rs, field_names):
    '''
    Read the result of a single query from the given string,
    returning a tuple of (record, remaining-string). If no complete
    record could be read, the first element of the tuple is None and
    the second element is the original imput string.
    '''

    rf = StringIO.StringIO(rs)

    def readline():
        l = rf.readline()
        if not l.endswith('\n'):
            raise EOFError()
        return l.strip()

    result = Result()

    try:
        l = readline()
        assert l.startswith('# BLAST')

        l = readline()
        assert l.startswith('# Query: ')
        query_str = l[len('# Query: '):].strip()
        if query_str:
            if ' ' in query_str:
                result.id, result.description = [
                    s.strip() for s in query_str.split(' ', 1)]
            else:
                result.id = query_str

        l = readline()
        assert l.startswith('# Database: ')

        l = readline()
        if l.startswith('# Fields: '):
            fns = l[len('# Fields: '):].split(', ')
            assert len(field_names) == len(fns)
            l = readline()

        assert l.endswith(' hits found')
        nhits = int(l[len('# '):-1 * len(' hits found')])

        while nhits > 0:
            l = readline()
            field_vals = l.split('\t')
            assert len(field_vals) == len(field_names)

            fields = dict(zip(field_names, field_vals))
            result.hits.append(Hit(fields))
            nhits -= 1

        return result, rf.read()
    except EOFError:
        return None, rs