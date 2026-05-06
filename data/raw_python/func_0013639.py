def getContinuous(self, referenceName=None, start=None, end=None):
        """
        Method passed to runSearchRequest to fulfill the request to
        yield continuous protocol objects that satisfy the given query.

        :param str referenceName: name of reference (ex: "chr1")
        :param start: castable to int, start position on reference
        :param end: castable to int, end position on reference
        :return: yields a protocol.Continuous at a time
        """
        bigWigReader = BigWigDataSource(self._filePath)
        for continuousObj in bigWigReader.bigWigToProtocol(
                                            referenceName, start, end):
            yield continuousObj