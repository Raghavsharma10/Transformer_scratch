def plotGrid(self, numLines=(5,5), lineWidth=1, colour="#777777"):
        """Plot NUMLINES[0] vertical gridlines and NUMLINES[1] horizontal gridlines,
        while keeping the initial axes bounds that were present upon its calling.
        Will not work for certain cases.
        """
        x1, x2, y1, y2 = mp.axis()
        ra1, dec0 = self.pixToSky(x1, y1)
        ra0, dec1 = self.pixToSky(x2, y2)

        xNum, yNum = numLines
        self.raRange, self.decRange  = self.getRaDecRanges(numLines)

        #import pdb; pdb.set_trace()
        #Guard against Ra of zero within the plot
        a1 = np.abs(ra1-ra0)
        a2 = np.abs( min(ra0, ra1) - (max(ra0, ra1) - 360))
        if a2 < a1:     #Then we straddle 360 degrees in RA
            if ra0 < ra1:
                ra1 -= 360
            else:
                ra0 -= 360


        #Draw lines of constant dec
        lwr = min(ra0, ra1)
        upr = max(ra0, ra1)
        stepX = round((upr-lwr) / float(xNum))
        ra_deg = np.arange(lwr - 3*stepX, upr + 3.5*stepX, 1, dtype=np.float)
        for dec in self.decRange:
            self.plotLine(ra_deg, dec, '-', color = colour, linewidth = lineWidth)


        #Draw lines of const ra
        lwr = min(dec0, dec1)
        upr = max(dec0, dec1)
        stepY = round((upr-lwr) / float(yNum))
        dec_deg = np.arange(dec0 - 3*stepY, dec1 + 3.5*stepY, 1, dtype=np.float)
        for ra in self.raRange:
            self.plotLine(ra, dec_deg, '-', color = colour, linewidth = lineWidth)

        mp.axis([x1, x2, y1, y2])