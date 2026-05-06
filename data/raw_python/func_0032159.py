def pickAChannel(self, ra_deg, dec_deg):
        """Returns the channel number closest to a given (ra, dec) coordinate.
        """
        # Could improve speed by doing this in the projection plane
        # instead of sky coords
        cRa = self.currentRaDec[:, 3]  # Ra of each channel corner
        cDec = self.currentRaDec[:, 4]  # dec of each channel corner

        dist = cRa * 0
        for i in range(len(dist)):
            dist[i] = gcircle.sphericalAngSep(cRa[i], cDec[i], ra_deg, dec_deg)

        i = np.argmin(dist)
        return self.currentRaDec[i, 2]