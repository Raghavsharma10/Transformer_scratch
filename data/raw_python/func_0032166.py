def plotSpacecraftYAxis(self, maptype=None):
        """Plot a line pointing in the direction of the spacecraft
        y-axis (i.e normal to the solar panel
        """

        if maptype is None:
            maptype=self.defaultMap
        #Plot direction of spacecraft +y axis. The subtraction of
        #90 degrees accounts for the different defintions of where
        #zero roll is.
        yAngle_deg = getSpacecraftRollAngleFromFovAngle(self.roll0_deg)
        yAngle_deg -=90

        a,d = gcircle.sphericalAngDestination(self.ra0_deg, self.dec0_deg, -yAngle_deg, 12.0)
        x0, y0 = maptype.skyToPix(self.ra0_deg, self.dec0_deg)
        x1, y1 = maptype.skyToPix(a, d)
        mp.plot([x0, x1], [y0, y1], 'k-')