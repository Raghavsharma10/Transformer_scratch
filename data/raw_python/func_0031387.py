def _alignmentToStr(self, result):
        """
        Make a textual representation of an alignment result.

        @param result: A C{dict}, as returned by C{self.createAlignment}.
        @return: A C{str} desription of a result. For every three lines the
            first and third contain the input sequences, possibly padded
            with '-'. The second contains '|' where the two sequences match,
            and ' ' where not.
            Format of the output is as follows:
            Cigar: (Cigar string)
            Evalue:
            Bitscore:
            Id1 Match start: (int) Match end: (int)
            Id2 Match start: (int) Match end: (int)
            Id1:  1 (seq) 50
            [lines to show matches]
            Id2:  1 (seq) 50
        """
        if result is None:
            return ('\nNo alignment between %s and %s\n' % (
                self.seq1ID, self.seq2ID))
        else:
            header = (
                '\nCigar string of aligned region: %s\n'
                '%s Match start: %d Match end: %d\n'
                '%s Match start: %d Match end: %d\n' %
                (result['cigar'],
                 self.seq1ID, result['sequence1Start'], result['sequence1End'],
                 self.seq2ID, result['sequence2Start'], result['sequence2End'])
            )
            text = '\n'.join(result['text'])

            return header + text