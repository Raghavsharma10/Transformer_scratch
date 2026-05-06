def adjustHSP(self, hsp):
        """
        Adjust the read and subject start and end offsets in an HSP.

        @param hsp: a L{dark.hsp.HSP} or L{dark.hsp.LSP} instance.
        """
        reduction = self._reductionForOffset(
            min(hsp.readStartInSubject, hsp.subjectStart))

        hsp.readEndInSubject = hsp.readEndInSubject - reduction
        hsp.readStartInSubject = hsp.readStartInSubject - reduction
        hsp.subjectEnd = hsp.subjectEnd - reduction
        hsp.subjectStart = hsp.subjectStart - reduction