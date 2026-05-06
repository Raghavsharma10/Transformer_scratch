def _input_as_seqs(self, data):
        """Format a list of seq as input.

        Parameters
        ----------
        data: list of strings
            Each string is a sequence to be aligned.

        Returns
        -------
        A temp file name that contains the sequences.

        See Also
        --------
        burrito.util.CommandLineApplication
        """
        lines = []
        for i, s in enumerate(data):
            # will number the sequences 1,2,3,etc.
            lines.append(''.join(['>', str(i+1)]))
            lines.append(s)
        return self._input_as_lines(lines)