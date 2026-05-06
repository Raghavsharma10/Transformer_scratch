def lookup(self, pathogenName, sampleName):
        """
        Look up a pathogen name, sample name combination and get its
        FASTA/FASTQ file name and unique read count.

        This method should be used instead of C{add} in situations where
        you want an exception to be raised if a pathogen/sample combination has
        not already been passed to C{add}.

        @param pathogenName: A C{str} pathogen name.
        @param sampleName: A C{str} sample name.
        @raise KeyError: If the pathogen name or sample name have not been
            seen, either individually or in combination.
        @return: A (C{str}, C{int}) tuple retrieved from self._readsFilenames
        """
        pathogenIndex = self._pathogens[pathogenName]
        sampleIndex = self._samples[sampleName]
        return self._readsFilenames[(pathogenIndex, sampleIndex)]