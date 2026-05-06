def toStr(self):
        """
        Produce a string representation of the pathogen summary.

        @return: A C{str} suitable for printing.
        """
        # Note that the string representation contains much less
        # information than the HTML summary. E.g., it does not contain the
        # unique (de-duplicated, by id) read count, since that is only computed
        # when we are making combined FASTA files of reads matching a
        # pathogen.
        readCountGetter = itemgetter('readCount')
        result = []
        append = result.append

        append(self._title())
        append('')

        for pathogenName in sorted(self.pathogenNames):
            samples = self.pathogenNames[pathogenName]
            sampleCount = len(samples)
            append('%s (in %d sample%s)' %
                   (pathogenName,
                    sampleCount, '' if sampleCount == 1 else 's'))
            for sampleName in sorted(samples):
                proteins = samples[sampleName]['proteins']
                proteinCount = len(proteins)
                totalReads = sum(readCountGetter(p) for p in proteins.values())
                append('  %s (%d protein%s, %d read%s)' %
                       (sampleName,
                        proteinCount, '' if proteinCount == 1 else 's',
                        totalReads, '' if totalReads == 1 else 's'))
                for proteinName in sorted(proteins):
                    append(
                        '    %(coverage).2f\t%(medianScore).2f\t'
                        '%(bestScore).2f\t%(readCount)4d\t%(hspCount)4d\t'
                        '%(index)3d\t%(proteinName)s'
                        % proteins[proteinName])
            append('')

        return '\n'.join(result)