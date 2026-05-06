def split_diff(old, new):
        """
        Returns a generator yielding the side-by-side diff of `old` and `new`).
        """
        return map(lambda l: l.rstrip(),
                   icdiff.ConsoleDiff(cols=COLUMNS).make_table(old.splitlines(), new.splitlines()))