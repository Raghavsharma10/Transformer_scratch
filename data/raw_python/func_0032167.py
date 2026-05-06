def plotChIds(self, maptype=None, modout=False):
        """Print the channel numbers on the plotting display

        Note:
        ---------
        This method will behave poorly if you are plotting in
        mixed projections. Because the channel vertex polygons
        are already projected using self.defaultMap, applying
        this function when plotting in a different reference frame
        may cause trouble.
        """
        if maptype is None:
            maptype = self.defaultMap

        polyList = self.getAllChannelsAsPolygons(maptype)
        for p in polyList:
            p.identifyModule(modout=modout)