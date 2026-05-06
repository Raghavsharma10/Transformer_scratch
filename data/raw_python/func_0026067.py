def scale_WCS(self,pixel_scale,retain=True):
        ''' Scale the WCS to a new pixel_scale. The 'retain' parameter
        [default value: True] controls whether or not to retain the original
        distortion solution in the CD matrix.
        '''
        _ratio = pixel_scale / self.pscale

        # Correct the size of the image and CRPIX values for scaled WCS
        self.naxis1 /= _ratio
        self.naxis2 /= _ratio
        self.crpix1 = self.naxis1/2.
        self.crpix2 = self.naxis2/2.

        if retain:
            # Correct the WCS while retaining original distortion information
            self.cd11 *= _ratio
            self.cd12 *= _ratio
            self.cd21 *= _ratio
            self.cd22 *= _ratio
        else:
            pscale = pixel_scale / 3600.
            self.cd11 = -pscale * N.cos(pa)
            self.cd12 = pscale * N.sin(pa)
            self.cd21 = self.cd12
            self.cd22 = -self.cd11

        # Now make sure that all derived values are really up-to-date based
        # on these changes
        self.update()