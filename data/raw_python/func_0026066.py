def updateWCS(self, pixel_scale=None, orient=None,refpos=None,refval=None,size=None):
        """
        Create a new CD Matrix from the absolute pixel scale
        and reference image orientation.
        """
        # Set up parameters necessary for updating WCS
        # Check to see if new value is provided,
        # If not, fall back on old value as the default

        _updateCD = no
        if orient is not None and orient != self.orient:
            pa = DEGTORAD(orient)
            self.orient = orient
            self._orient_lin = orient
            _updateCD = yes
        else:
            # In case only pixel_scale was specified
            pa = DEGTORAD(self.orient)

        if pixel_scale is not None and pixel_scale != self.pscale:
            _ratio = pixel_scale / self.pscale
            self.pscale = pixel_scale
            _updateCD = yes
        else:
            # In case, only orient was specified
            pixel_scale = self.pscale
            _ratio = None

        # If a new plate scale was given,
        # the default size should be revised accordingly
        # along with the default reference pixel position.
        # Added 31 Mar 03, WJH.
        if _ratio is not None:
            self.naxis1 /= _ratio
            self.naxis2 /= _ratio
            self.crpix1 = self.naxis1/2.
            self.crpix2 = self.naxis2/2.

        # However, if the user provides a given size,
        # set it to use that no matter what.
        if size is not None:
            self.naxis1 = size[0]
            self.naxis2 = size[1]

        # Insure that naxis1,2 always return as integer values.
        self.naxis1 = int(self.naxis1)
        self.naxis2 = int(self.naxis2)

        if refpos is not None:
            self.crpix1 = refpos[0]
            self.crpix2 = refpos[1]
        if self.crpix1 is None:
            self.crpix1 = self.naxis1/2.
            self.crpix2 = self.naxis2/2.

        if refval is not None:
            self.crval1 = refval[0]
            self.crval2 = refval[1]

        # Reset WCS info now...
        if _updateCD:
            # Only update this should the pscale or orientation change...
            pscale = pixel_scale / 3600.

            self.cd11 = -pscale * N.cos(pa)
            self.cd12 = pscale * N.sin(pa)
            self.cd21 = self.cd12
            self.cd22 = -self.cd11

        # Now make sure that all derived values are really up-to-date based
        # on these changes
        self.update()