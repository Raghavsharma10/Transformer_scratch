def plotOutline(self, maptype=None, colour='#AAAAAA', **kwargs):
        """Plot an outline of the FOV.
        """

        if maptype is None:
            maptype=self.defaultMap

        xarr = []
        yarr = []
        radec = self.currentRaDec
        for ch in [20,4,11,28,32, 71,68, 84, 75, 60, 56, 15 ]:
            idx = np.where(radec[:,2].astype(np.int) == ch)[0]
            idx = idx[0]    #Take on the first one
            x, y = maptype.skyToPix(radec[idx][3], radec[idx][4])
            xarr.append(x)
            yarr.append(y)

        verts = np.empty( (len(xarr), 2))
        verts[:,0] = xarr
        verts[:,1] = yarr

        #There are two ways to specify line colour
        ec = kwargs.pop('ec', "none")
        ec = kwargs.pop('edgecolor', ec)
        p = matplotlib.patches.Polygon(verts, fill=True, ec=ec, fc=colour, **kwargs)
        mp.gca().add_patch(p)