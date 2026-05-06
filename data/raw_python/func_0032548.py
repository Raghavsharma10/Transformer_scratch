def iter_tsv(input_stream, cols=None, encoding='utf-8'):
    """
    If a tuple is given in cols, use the elements as names to construct
    a namedtuple.

    Columns can be marked as ignored by using ``X`` or ``0`` as column name.

    Example (ignore the first four columns of a five column TSV):

    ::

        def run(self):
            with self.input().open() as handle:
                for row in handle.iter_tsv(cols=('X', 'X', 'X', 'X', 'iln')):
                    print(row.iln)
    """
    if cols:
        cols = [c if not c in ('x', 'X', 0, None) else random_string(length=5)
                for c in cols]
        Record = collections.namedtuple('Record', cols)
        for line in input_stream:
            yield Record._make(line.decode(encoding).rstrip('\n').split('\t'))
    else:
        for line in input_stream:
            yield tuple(line.decode(encoding).rstrip('\n').split('\t'))