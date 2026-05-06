def plotPointing(self, maptype=None, colour='b', mod3='r', showOuts=True, **kwargs):
        """Plot the FOV
        """

        if maptype is None:
            maptype=self.defaultMap

        radec = self.currentRaDec
        for ch in radec[:,2][::4]:

            idx = np.where(radec[:,2].astype(np.int) == ch)[0]
            idx = np.append(idx, idx[0])  #% points to draw a box

            c = colour
            if ch in self.brokenChannels:
                c = mod3
            maptype.plot(radec[idx, 3], radec[idx, 4], '-', color=c, **kwargs)
            #Show the origin of the col and row coords for this ch
            if showOuts:
                maptype.plot(radec[idx[0], 3], radec[idx[0],4], 'o', color=c)