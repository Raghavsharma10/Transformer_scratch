def add(self, pathogenName, sampleName):
        """
        Add a (pathogen name, sample name) combination and get its FASTA/FASTQ
        file name and unique read count. Write the FASTA/FASTQ file if it does
        not already exist. Save the unique read count into
        C{self._proteinGrouper}.

        @param pathogenName: A C{str} pathogen name.
        @param sampleName: A C{str} sample name.
        @return: A C{str} giving the FASTA/FASTQ file name holding all the
            reads (without duplicates, by id) from the sample that matched the
            proteins in the given pathogen.
        """
        pathogenIndex = self._pathogens.setdefault(pathogenName,
                                                   len(self._pathogens))
        sampleIndex = self._samples.setdefault(sampleName, len(self._samples))

        try:
            return self._readsFilenames[(pathogenIndex, sampleIndex)]
        except KeyError:
            reads = Reads()
            for proteinMatch in self._proteinGrouper.pathogenNames[
                    pathogenName][sampleName]['proteins'].values():
                for read in self._readsClass(proteinMatch['readsFilename']):
                    reads.add(read)
            saveFilename = join(
                proteinMatch['outDir'],
                'pathogen-%d-sample-%d.%s' % (pathogenIndex, sampleIndex,
                                              self._format))
            reads.filter(removeDuplicatesById=True)
            nReads = reads.save(saveFilename, format_=self._format)
            # Save the unique read count into self._proteinGrouper
            self._proteinGrouper.pathogenNames[
                pathogenName][sampleName]['uniqueReadCount'] = nReads
            self._readsFilenames[(pathogenIndex, sampleIndex)] = saveFilename
            return saveFilename