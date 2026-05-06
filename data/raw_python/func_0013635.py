def readValuesPyBigWig(self, reference, start, end):
        """
        Use pyBigWig package to read a BigWig file for the
        given range and return a protocol object.

        pyBigWig returns an array of values that fill the query range.
        Not sure if it is possible to get the step and span.

        This method trims NaN values from the start and end.

        pyBigWig throws an exception if end is outside of the
        reference range. This function checks the query range
        and throws its own exceptions to avoid the ones thrown
        by pyBigWig.
        """
        if not self.checkReference(reference):
            raise exceptions.ReferenceNameNotFoundException(reference)
        if start < 0:
            start = 0
        bw = pyBigWig.open(self._sourceFile)
        referenceLen = bw.chroms(reference)
        if referenceLen is None:
            raise exceptions.ReferenceNameNotFoundException(reference)
        if end > referenceLen:
            end = referenceLen
        if start >= end:
            raise exceptions.ReferenceRangeErrorException(
                reference, start, end)

        data = protocol.Continuous()
        curStart = start
        curEnd = curStart + self._INCREMENT
        while curStart < end:
            if curEnd > end:
                curEnd = end
            for i, val in enumerate(bw.values(reference, curStart, curEnd)):
                if not math.isnan(val):
                    if len(data.values) == 0:
                        data.start = curStart + i
                    data.values.append(val)
                    if len(data.values) == self._MAX_VALUES:
                        yield data
                        data = protocol.Continuous()
                elif len(data.values) > 0:
                    # data.values.append(float('NaN'))
                    yield data
                    data = protocol.Continuous()
            curStart = curEnd
            curEnd = curStart + self._INCREMENT

        bw.close()
        if len(data.values) > 0:
            yield data