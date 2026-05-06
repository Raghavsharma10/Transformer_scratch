def convertLocationHgvsP(self, hgvsp):
        """
        Accepts an annotation in HGVS notation and returns
        an AlleleLocation with populated fields.
        :param hgvsp:
        :return: protocol.AlleleLocation
        """
        if isUnspecified(hgvsp):
            return None
        match = re.match(".*p.(\D+)(\d+)(\D+)", hgvsp, flags=re.UNICODE)
        if match is not None:
            allLoc = self._createGaAlleleLocation()
            allLoc.reference_sequence = match.group(1)
            allLoc.start = int(match.group(2)) - 1
            allLoc.alternate_sequence = match.group(3)
            return allLoc
        return None