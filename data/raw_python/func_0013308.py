def convertLocation(self, pos):
        """
        Accepts a position string (start/length) and returns
        a GA4GH AlleleLocation with populated fields.
        :param pos:
        :return: protocol.AlleleLocation
        """
        if isUnspecified(pos):
            return None
        coordLen = pos.split('/')
        if len(coordLen) > 1:
            allLoc = self._createGaAlleleLocation()
            allLoc.start = int(coordLen[0]) - 1
            return allLoc
        return None