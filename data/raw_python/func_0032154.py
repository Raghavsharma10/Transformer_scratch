def getChannelColRowList(self, ra, dec, wantZeroOffset=False,
                         allowIllegalReturnValues=True):
        """similar to getChannelColRow() but takes lists as input"""
        try:
            ch = self.pickAChannelList(ra, dec)
        except ValueError:
            logger.warning("WARN: %.7f %.7f not on any channel" % (ra, dec))
            return (0, 0, 0)

        col = np.zeros(len(ch))
        row = np.zeros(len(ch))
        for channel in set(ch):
            mask = (ch == channel)
            col[mask], row[mask] = self.getColRowWithinChannelList(ra[mask], dec[mask], channel, 
                                                        wantZeroOffset, allowIllegalReturnValues)
        return (ch, col, row)