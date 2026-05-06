def save(self, filename, format_='fasta'):
        """
        Write the reads to C{filename} in the requested format.

        @param filename: Either a C{str} file name to save into (the file will
            be overwritten) or an open file descriptor (e.g., sys.stdout).
        @param format_: A C{str} format to save as, either 'fasta', 'fastq' or
            'fasta-ss'.
        @raise ValueError: if C{format_} is 'fastq' and a read with no quality
            is present, or if an unknown format is requested.
        @return: An C{int} giving the number of reads in C{self}.
        """
        format_ = format_.lower()
        count = 0

        if isinstance(filename, str):
            try:
                with open(filename, 'w') as fp:
                    for read in self:
                        fp.write(read.toString(format_))
                        count += 1
            except ValueError:
                unlink(filename)
                raise
        else:
            # We have a file-like object.
            for read in self:
                filename.write(read.toString(format_))
                count += 1
        return count