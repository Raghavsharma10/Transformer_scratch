def getChannelColRow(self, ra, dec, wantZeroOffset=False,
                         allowIllegalReturnValues=True):
        """Returns (channel, column, row) given an (ra, dec) coordinate.

        Returns (0, 0, 0) or a ValueError if the coordinate is not on silicon.
        """
        try:
            ch = self.pickAChannel(ra, dec)
        except ValueError:
            logger.warning("WARN: %.7f %.7f not on any channel" % (ra, dec))
            return (0, 0, 0)

        col, row = self.getColRowWithinChannel(ra, dec, ch, wantZeroOffset,
                                               allowIllegalReturnValues)
        return (ch, col, row)