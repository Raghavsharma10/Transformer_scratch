def iter(self):
        """
        Iterate over the sequences in the files in self.files_, yielding each
        as an instance of the desired read class.
        """
        count = 0
        for _file in self._files:
            with asHandle(_file) as fp:
                # Duplicate some code here so as not to test
                # self._upperCase in the loop.
                if self._upperCase:
                    for seq in SeqIO.parse(fp, 'fasta'):
                        read = self._readClass(seq.description,
                                               str(seq.seq.upper()))
                        yield read
                        count += 1
                else:
                    for seq in SeqIO.parse(fp, 'fasta'):
                        read = self._readClass(seq.description, str(seq.seq))
                        yield read
                        count += 1