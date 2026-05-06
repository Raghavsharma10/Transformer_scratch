def createWcsHDU(self):
        """ Generate a WCS header object that can be used to
            populate a reference WCS HDU.
        """
        hdu = fits.ImageHDU()
        hdu.header['EXTNAME'] = 'WCS'
        hdu.header['EXTVER'] = 1
        # Now, update original image size information
        hdu.header['WCSAXES'] = (2, "number of World Coordinate System axes")
        hdu.header['NPIX1'] = (self.naxis1, "Length of array axis 1")
        hdu.header['NPIX2'] = (self.naxis2, "Length of array axis 2")
        hdu.header['PIXVALUE'] = (0.0, "values of pixels in array")

        # Write out values to header...
        excluded_keys = ['naxis1','naxis2']
        for key in self.wcskeys:
            _dkey = self.wcstrans[key]
            if _dkey not in excluded_keys:
                hdu.header[key] = self.__dict__[_dkey]


        return hdu