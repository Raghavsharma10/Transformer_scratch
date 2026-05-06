def getColRowWithinChannel(self, ra, dec, ch, wantZeroOffset=False,
                               allowIllegalReturnValues=True):
        """Returns (col, row) given a (ra, dec) coordinate and channel number.
        """
        # How close is a given ra/dec to the origin of a KeplerModule?
        x, y = self.defaultMap.skyToPix(ra, dec)
        kepModule = self.getChannelAsPolygon(ch)
        r = np.array([x[0],y[0]]) - kepModule.polygon[0, :]

        v1 = kepModule.polygon[1, :] - kepModule.polygon[0, :]
        v3 = kepModule.polygon[3, :] - kepModule.polygon[0, :]

        # Divide by |v|^2 because you're normalising v and r
        colFrac = np.dot(r, v1) / np.linalg.norm(v1)**2
        rowFrac = np.dot(r, v3) / np.linalg.norm(v3)**2

        # This is where it gets a little hairy. The channel "corners"
        # supplied to me actually represent points 5x5 pixels inside
        # the science array. Which isn't what you'd expect.
        # These magic numbers are the pixel numbers of the corner
        # edges given in fov.txt
        col = colFrac*(1106-17) + 17
        row = rowFrac*(1038-25) + 25

        if not allowIllegalReturnValues:
            if not self.colRowIsOnSciencePixel(col, row):
                msg = "Request position %7f %.7f " % (ra, dec)
                msg += "does not lie on science pixels for channel %i " % (ch)
                msg += "[ %.1f %.1f]" % (col, row)
                raise ValueError(msg)

        # Convert from zero-offset to one-offset coords
        if not wantZeroOffset:
            col += 1
            row += 1

        return (col, row)