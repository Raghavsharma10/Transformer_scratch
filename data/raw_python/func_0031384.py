def _initialise(self):
        """
        Initialises table with dictionary.
        """
        d = {'score': 0, 'pointer': None, 'ins': 0, 'del': 0}
        cols = len(self.seq1Seq) + 1
        rows = len(self.seq2Seq) + 1
        # Note that this puts a ref to the same dict (d) into each cell of
        # the table. Hopefully that is what was intended. Eyeballing the
        # code below that uses the table it looks like table entries are
        # entirely replaced, so this seems ok. Terry.
        table = [[d for _ in range(cols)] for _ in range(rows)]
        return table