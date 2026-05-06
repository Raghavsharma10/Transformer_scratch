def populateFromHeader(self, readGroupHeader):
        """
        Populate the instance variables using the specified SAM header.
        """
        self._sampleName = readGroupHeader.get('SM', None)
        self._description = readGroupHeader.get('DS', None)
        if 'PI' in readGroupHeader:
            self._predictedInsertSize = int(readGroupHeader['PI'])
        self._instrumentModel = readGroupHeader.get('PL', None)
        self._sequencingCenter = readGroupHeader.get('CN', None)
        self._experimentDescription = readGroupHeader.get('DS', None)
        self._library = readGroupHeader.get('LB', None)
        self._platformUnit = readGroupHeader.get('PU', None)
        self._runTime = readGroupHeader.get('DT', None)