def line(self, idx):
        """Return the i'th program line.

        :param i: The i'th program line.

        """
        # TODO: We should parse the response properly.
        return self._query(('PGM?', [Integer, Integer], String), self.idx, idx)