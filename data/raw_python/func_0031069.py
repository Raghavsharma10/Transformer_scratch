def addFile(self, filename, fp):
        """
        Read and record protein information for a sample.

        @param filename: A C{str} file name.
        @param fp: An open file pointer to read the file's data from.
        @raise ValueError: If information for a pathogen/protein/sample
            combination is given more than once.
        """
        if self._sampleName:
            sampleName = self._sampleName
        elif self._sampleNameRegex:
            match = self._sampleNameRegex.search(filename)
            if match:
                sampleName = match.group(1)
            else:
                sampleName = filename
        else:
            sampleName = filename

        outDir = join(dirname(filename), self._assetDir)

        self.sampleNames[sampleName] = join(outDir, 'index.html')

        for index, proteinLine in enumerate(fp):
            proteinLine = proteinLine[:-1]
            (coverage, medianScore, bestScore, readCount, hspCount,
             proteinLength, names) = proteinLine.split(None, 6)

            proteinName, pathogenName = splitNames(names)

            if pathogenName not in self.pathogenNames:
                self.pathogenNames[pathogenName] = {}

            if sampleName not in self.pathogenNames[pathogenName]:
                self.pathogenNames[pathogenName][sampleName] = {
                    'proteins': {},
                    'uniqueReadCount': None,
                }

            proteins = self.pathogenNames[pathogenName][sampleName]['proteins']

            # We should only receive one line of information for a given
            # pathogen/sample/protein combination.
            if proteinName in proteins:
                raise ValueError(
                    'Protein %r already seen for pathogen %r sample %r.' %
                    (proteinName, pathogenName, sampleName))

            readsFilename = join(outDir, '%d.%s' % (index, self._format))

            proteins[proteinName] = {
                'bestScore': float(bestScore),
                'bluePlotFilename': join(outDir, '%d.png' % index),
                'coverage': float(coverage),
                'readsFilename': readsFilename,
                'hspCount': int(hspCount),
                'index': index,
                'medianScore': float(medianScore),
                'outDir': outDir,
                'proteinLength': int(proteinLength),
                'proteinName': proteinName,
                'proteinURL': NCBISequenceLinkURL(proteinName),
                'readCount': int(readCount),
            }

            if self._saveReadLengths:
                readsClass = (FastaReads if self._format == 'fasta'
                              else FastqReads)
                proteins[proteinName]['readLengths'] = tuple(
                    len(read) for read in readsClass(readsFilename))