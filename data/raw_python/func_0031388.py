def createAlignment(self, resultFormat=dict):
        """
        Run the alignment algorithm.

        @param resultFormat: Either C{dict} or C{str}, giving the desired
            result format.
        @return: If C{resultFormat} is C{dict}, a C{dict} containing
            information about the match (or C{None}) if there is no match.
            If C{resultFormat} is C{str}, a C{str} containing a readable
            version of the match info (see _alignmentToStr above for the exact
            format).
        """
        table = self._initialise()
        alignment = self._fillAndTraceback(table)
        output = alignment[0]
        if output[0] == '' or output[2] == '':
            result = None
        else:
            indexes = alignment[1]
            result = {
                'cigar': self._cigarString(output),
                'sequence1Start': indexes['min_col'],
                'sequence1End': indexes['max_col'],
                'sequence2Start': indexes['min_row'],
                'sequence2End': indexes['max_row'],
                'text': self._formatAlignment(output, indexes),
            }

        return self._alignmentToStr(result) if resultFormat is str else result