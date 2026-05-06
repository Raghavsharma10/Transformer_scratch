def _getReadAlignments(
            self, reference, start, end, readGroupSet, readGroup):
        """
        Returns an iterator over the specified reads
        """
        # TODO If reference is None, return against all references,
        # including unmapped reads.
        samFile = self.getFileHandle(self._dataUrl)
        referenceName = reference.getLocalId().encode()
        # TODO deal with errors from htslib
        start, end = self.sanitizeAlignmentFileFetch(start, end)
        readAlignments = samFile.fetch(referenceName, start, end)
        for readAlignment in readAlignments:
            tags = dict(readAlignment.tags)
            if readGroup is None:
                if 'RG' in tags:
                    alignmentReadGroupLocalId = tags['RG']
                    readGroupCompoundId = datamodel.ReadGroupCompoundId(
                        readGroupSet.getCompoundId(),
                        str(alignmentReadGroupLocalId))
                yield self.convertReadAlignment(
                    readAlignment, readGroupSet, str(readGroupCompoundId))
            else:
                if self._filterReads:
                    if 'RG' in tags and tags['RG'] == self._localId:
                        yield self.convertReadAlignment(
                            readAlignment, readGroupSet,
                            str(readGroup.getCompoundId()))
                else:
                    yield self.convertReadAlignment(
                        readAlignment, readGroupSet,
                        str(readGroup.getCompoundId()))