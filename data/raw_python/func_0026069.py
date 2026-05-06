def rd2xy(self,skypos,hour=no):
        """
        This method would use the WCS keywords to compute the XY position
        from a given RA/Dec tuple (in deg).

        NOTE: Investigate how to let this function accept arrays as well
        as single positions. WJH 27Mar03

        """
        if self.ctype1.find('TAN') < 0 or self.ctype2.find('TAN') < 0:
            print('RD2XY only supported for TAN projections.')
            raise TypeError

        det = self.cd11*self.cd22 - self.cd12*self.cd21

        if det == 0.0:
            raise ArithmeticError("singular CD matrix!")

        cdinv11 = self.cd22 / det
        cdinv12 = -self.cd12 / det
        cdinv21 = -self.cd21 / det
        cdinv22 = self.cd11 / det

        # translate (ra, dec) to (x, y)

        ra0 = DEGTORAD(self.crval1)
        dec0 = DEGTORAD(self.crval2)
        if hour:
            skypos[0] = skypos[0] * 15.
        ra = DEGTORAD(skypos[0])
        dec = DEGTORAD(skypos[1])

        bottom = float(N.sin(dec)*N.sin(dec0) + N.cos(dec)*N.cos(dec0)*N.cos(ra-ra0))
        if bottom == 0.0:
            raise ArithmeticError("Unreasonable RA/Dec range!")

        xi = RADTODEG((N.cos(dec) * N.sin(ra-ra0) / bottom))
        eta = RADTODEG((N.sin(dec)*N.cos(dec0) - N.cos(dec)*N.sin(dec0)*N.cos(ra-ra0)) / bottom)

        x = cdinv11 * xi + cdinv12 * eta + self.crpix1
        y = cdinv21 * xi + cdinv22 * eta + self.crpix2

        return x,y