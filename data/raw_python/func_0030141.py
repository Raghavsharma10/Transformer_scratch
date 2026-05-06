def select(self, predicate=None, headers=None):
        """
        Select rows from the reader using a predicate to select rows and and itemgetter to return a
        subset of elements
        :param predicate: If defined, a callable that is called for each row, and if it returns true, the
        row is included in the output.
        :param headers: If defined, a list or tuple of header names to return from each row
        :return: iterable of results

        WARNING: This routine works from the reader iterator, which returns RowProxy objects. RowProxy objects
        are reused, so if you construct a list directly from the output from this method, the list will have
        multiple copies of a single RowProxy, which will have as an inner row the last result row. If you will
        be directly constructing a list, use a getter that extracts the inner row, or which converts the RowProxy
        to a dict:

            list(s.datafile.select(lambda r: r.stusab == 'CA', lambda r: r.dict ))

        """

        # FIXME; in Python 3, use yield from
        with self.reader as r:
            for row in r.select(predicate, headers):
                yield row