def referenceLengths(self):
        """
        Get the lengths of wanted references.

        @raise UnknownReference: If a reference id is not present in the
            SAM/BAM file.
        @return: A C{dict} of C{str} reference id to C{int} length with a key
            for each reference id in C{self.referenceIds} or for all references
            if C{self.referenceIds} is C{None}.
        """
        result = {}
        with samfile(self.filename) as sam:
            if self.referenceIds:
                for referenceId in self.referenceIds:
                    tid = sam.get_tid(referenceId)
                    if tid == -1:
                        raise UnknownReference(
                            'Reference %r is not present in the SAM/BAM file.'
                            % referenceId)
                    else:
                        result[referenceId] = sam.lengths[tid]
            else:
                result = dict(zip(sam.references, sam.lengths))

        return result