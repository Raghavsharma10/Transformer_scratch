def getRaDecRanges(self, numLines):
        """Pick suitable values for ra and dec ticks

        Used by plotGrid and labelAxes
        """
        x1, x2, y1, y2 = mp.axis()

        ra0, dec0 = self.pixToSky(x1, y1)
        ra1, dec1 = self.pixToSky(x2, y2)

        #Deal with the case where ra range straddles 0.
        #Different code for case where ra increases left to right, or decreases.
        if self.isPositiveMap():
            if ra1 < ra0:
                ra1 += 360
        else:
            if ra0 < ra1:
                ra0 += 360

        raMid = .5*(ra0+ra1)
        decMid = .5*(dec0+dec1)


        xNum, yNum = numLines
        stepX = round((ra1 - ra0) / xNum)
        stepY = round((dec1 - dec0) / yNum)

        rangeX = stepX * (xNum - 1)
        rangeY = stepY * (yNum - 1)

        raStart = np.round(raMid - rangeX/2.)
        decStart = np.round(decMid - rangeY/2.)

        raRange = np.arange(raStart, raStart + stepX*xNum, stepX)
        decRange = np.arange(decStart, decStart + stepY*yNum, stepY)
        raRange = np.fmod(raRange, 360.)

        return raRange, decRange