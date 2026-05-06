def toString(self, format_='fasta-ss', structureSuffix=':structure'):
        """
        Convert the read to a string in PDB format (sequence & structure). This
        consists of two FASTA records, one for the sequence then one for the
        structure.

        @param format_: Either 'fasta-ss' or 'fasta'. In the former case, the
            structure information is returned. Otherwise, plain FASTA is
            returned.
        @param structureSuffix: The C{str} suffix to append to the read id
            for the second FASTA record, containing the structure information.
        @raise ValueError: If C{format_} is not 'fasta'.
        @return: A C{str} representing the read sequence and structure in PDB
            FASTA format.
        """
        if format_ == 'fasta-ss':
            return '>%s\n%s\n>%s%s\n%s\n' % (
                self.id, self.sequence, self.id, structureSuffix,
                self.structure)
        else:
            if six.PY3:
                return super().toString(format_=format_)
            else:
                return AARead.toString(self, format_=format_)