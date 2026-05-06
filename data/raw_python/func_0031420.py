def tabSeparatedSummary(self, sortOn=None):
        """
        Summarize all the alignments for this title as multi-line string with
        TAB-separated values on each line.

        @param sortOn: A C{str} attribute to sort titles on. One of 'length',
            'maxScore', 'medianScore', 'readCount', or 'title'.
        @raise ValueError: If an unknown C{sortOn} value is given.
        @return: A newline-separated C{str}, each line with a summary of a
            title. Each summary line is TAB-separated.
        """
        # The order of the fields returned here is somewhat arbitrary. The
        # subject titles are last because they are so variable in length.
        # Putting them last makes it more likely that the initial columns in
        # printed output will be easier to read down.
        #
        # Note that post-processing scripts will be relying on the field
        # ordering here.  So you can't just add fields. It's probably safe
        # to add them at the end, but be careful / think.
        #
        # A TAB-separated file can easily be read by awk using e.g.,
        # awk 'BEGIN {FS = "\t"} ...'

        result = []
        for titleSummary in self.summary(sortOn):
            result.append('\t'.join([
                '%(coverage)f',
                '%(medianScore)f',
                '%(bestScore)f',
                '%(readCount)d',
                '%(hspCount)d',
                '%(subjectLength)d',
                '%(subjectTitle)s',
            ]) % titleSummary)
        return '\n'.join(result)