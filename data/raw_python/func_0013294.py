def _checkCallSetIds(self, variantFile):
        """
        Checks callSetIds for consistency
        """
        if len(self._callSetIdMap) > 0:
            callSetIds = set([
                self.getCallSetId(sample)
                for sample in variantFile.header.samples])
            if callSetIds != set(self._callSetIdMap.keys()):
                raise exceptions.InconsistentCallSetIdException(
                    variantFile.filename)