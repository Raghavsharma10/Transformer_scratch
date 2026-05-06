def residueCounts(self, convertCaseTo='upper'):
        """
        Count residue frequencies at all sequence locations matched by reads.

        @param convertCaseTo: A C{str}, 'upper', 'lower', or 'none'.
            If 'none', case will not be converted (both the upper and lower
            case string of a residue will be present in the result if they are
            present in the read - usually due to low complexity masking).
        @return: A C{dict} whose keys are C{int} offsets into the title
            sequence and whose values are C{Counters} with the residue as keys
            and the count of that residue at that location as values.
        """
        if convertCaseTo == 'none':
            def convert(x):
                return x
        elif convertCaseTo == 'lower':
            convert = str.lower
        elif convertCaseTo == 'upper':
            convert = str.upper
        else:
            raise ValueError(
                "convertCaseTo must be one of 'none', 'lower', or 'upper'")

        counts = defaultdict(Counter)

        for titleAlignment in self:
            read = titleAlignment.read
            for hsp in titleAlignment.hsps:
                for (subjectOffset, residue, inMatch) in read.walkHSP(hsp):
                    counts[subjectOffset][convert(residue)] += 1

        return counts