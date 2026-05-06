def _computeUniqueReadCounts(self):
        """
        Add all pathogen / sample combinations to self.pathogenSampleFiles.

        This will make all de-duplicated (by id) FASTA/FASTQ files and store
        the number of de-duplicated reads into C{self.pathogenNames}.
        """
        for pathogenName, samples in self.pathogenNames.items():
            for sampleName in samples:
                self.pathogenSampleFiles.add(pathogenName, sampleName)