def xy2rd(self,pos):
        """
        This method would apply the WCS keywords to a position to
        generate a new sky position.

        The algorithm comes directly from 'imgtools.xy2rd'

        translate (x,y) to (ra, dec)
        """
        if self.ctype1.find('TAN') < 0 or self.ctype2.find('TAN') < 0:
            print('XY2RD only supported for TAN projections.')
            raise TypeError

        if isinstance(pos,N.ndarray):
            # If we are working with an array of positions,
            # point to just X and Y values
            posx = pos[:,0]
            posy = pos[:,1]
        else:
            # Otherwise, we are working with a single X,Y tuple
            posx = pos[0]
            posy = pos[1]

        xi = self.cd11 * (posx - self.crpix1) + self.cd12 * (posy - self.crpix2)
        eta = self.cd21 * (posx - self.crpix1) + self.cd22 * (posy - self.crpix2)

        xi = DEGTORAD(xi)
        eta = DEGTORAD(eta)
        ra0 = DEGTORAD(self.crval1)
        dec0 = DEGTORAD(self.crval2)

        ra = N.arctan((xi / (N.cos(dec0)-eta*N.sin(dec0)))) + ra0
        dec = N.arctan( ((eta*N.cos(dec0)+N.sin(dec0)) /
                (N.sqrt((N.cos(dec0)-eta*N.sin(dec0))**2 + xi**2))) )

        ra = RADTODEG(ra)
        dec = RADTODEG(dec)
        ra = DIVMOD(ra, 360.)

        # Otherwise, just return the RA,Dec tuple.
        return ra,dec