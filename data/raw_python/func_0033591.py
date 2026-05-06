def _get_result_paths(self, data):
        """Gets the results for a run of bwa index.

        bwa index outputs 5 files when the index is created. The filename
        prefix will be the same as the input fasta, unless overridden with
        the -p option, and the 5 extensions are listed below:

        .amb
        .ann
        .bwt
        .pac
        .sa

        and these extentions (including the period) are the keys to the
        dictionary that is returned.
        """

        # determine the names of the files. The name will be the same as the
        # input fasta file unless overridden with the -p option
        if self.Parameters['-p'].isOn():
            prefix = self.Parameters['-p'].Value
        else:
            prefix = data['fasta_in']

        # the 5 output file suffixes
        suffixes = ['.amb', '.ann', '.bwt', '.pac', '.sa']
        out_files = {}
        for suffix in suffixes:
            out_files[suffix] = ResultPath(prefix + suffix, IsWritten=True)

        return out_files