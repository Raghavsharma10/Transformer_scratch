def convertLocationHgvsC(self, hgvsc):
        """
        Accepts an annotation in HGVS notation and returns
        an AlleleLocation with populated fields.
        :param hgvsc:
        :return:
        """
        if isUnspecified(hgvsc):
            return None
        match = re.match(".*c.(\d+)(\D+)>(\D+)", hgvsc)
        if match:
            pos = int(match.group(1))
            if pos > 0:
                allLoc = self._createGaAlleleLocation()
                allLoc.start = pos - 1
                allLoc.reference_sequence = match.group(2)
                allLoc.alternate_sequence = match.group(3)
                return allLoc
        return None