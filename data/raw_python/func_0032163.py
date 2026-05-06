def getAllChannelsAsPolygons(self, maptype=None):
        """Return slew the telescope and return the corners of the modules
        as Polygon objects.

        If a projection is supplied, the ras and
        decs are mapped onto x, y using that projection
        """
        polyList = []
        for ch in self.origin[:, 2]:
            poly = self.getChannelAsPolygon(ch, maptype)
            polyList.append(poly)
        return polyList