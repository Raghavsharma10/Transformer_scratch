def createBamHeader(self, baseHeader):
        """
        Creates a new bam header based on the specified header from the
        parent BAM file.
        """
        header = dict(baseHeader)
        newSequences = []
        for index, referenceInfo in enumerate(header['SQ']):
            if index < self.numChromosomes:
                referenceName = referenceInfo['SN']
                # The sequence dictionary in the BAM file has to match up
                # with the sequence ids in the data, so we must be sure
                # that these still match up.
                assert referenceName == self.chromosomes[index]
                newReferenceInfo = {
                    'AS': self.referenceSetName,
                    'SN': referenceName,
                    'LN': 0,  # FIXME
                    'UR': 'http://example.com',
                    'M5': 'dbb6e8ece0b5de29da56601613007c2a',  # FIXME
                    'SP': 'Human'
                }
                newSequences.append(newReferenceInfo)
        header['SQ'] = newSequences
        return header