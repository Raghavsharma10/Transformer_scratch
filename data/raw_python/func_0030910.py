def _writeFASTA(self, i, image):
        """
        Write a FASTA file containing the set of reads that hit a sequence.

        @param i: The number of the image in self._images.
        @param image: A member of self._images.
        @return: A C{str}, either 'fasta' or 'fastq' indicating the format
            of the reads in C{self._titlesAlignments}.
        """
        if isinstance(self._titlesAlignments.readsAlignments.reads,
                      FastqReads):
            format_ = 'fastq'
        else:
            format_ = 'fasta'
        filename = '%s/%d.%s' % (self._outputDir, i, format_)
        titleAlignments = self._titlesAlignments[image['title']]
        with open(filename, 'w') as fp:
            for titleAlignment in titleAlignments:
                fp.write(titleAlignment.read.toString(format_))
        return format_