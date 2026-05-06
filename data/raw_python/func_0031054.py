def iter(self):
        """
        Iterate over the sequences in the files in self.files_, yielding each
        as an instance of the desired read class.
        """
        if self._upperCase:
            for id_ in self._fasta:
                yield self._readClass(id_, str(self._fasta[id_]).upper())
        else:
            for id_ in self._fasta:
                yield self._readClass(id_, str(self._fasta[id_]))