def _aln_filename(self,prefix):
        """Return name of file containing the alignment

        prefix -- str, prefix of alignment file.
        """
        if self.Parameters['-outfile'].isOn():
            aln_filename = self._absolute(self.Parameters['-outfile'].Value)
        else:
            aln_filename = prefix + self._suffix()
        return aln_filename