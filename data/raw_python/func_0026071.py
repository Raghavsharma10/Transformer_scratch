def write(self,fitsname=None,wcs=None,archive=True,overwrite=False,quiet=True):
        """
        Write out the values of the WCS keywords to the
        specified image.

        If it is a GEIS image and 'fitsname' has been provided,
        it will automatically make a multi-extension
        FITS copy of the GEIS and update that file. Otherwise, it
        throw an Exception if the user attempts to directly update
        a GEIS image header.

        If archive=True, also write out archived WCS keyword values to file.
        If overwrite=True, replace archived WCS values in file with new values.

        If a WCSObject is passed through the 'wcs' keyword, then the WCS keywords
        of this object are copied to the header of the image to be updated. A use case
        fo rthis is updating the WCS of a WFPC2 data quality (_c1h.fits) file
        in order to be in sync with the science (_c0h.fits) file.

        """
        ## Start by making sure all derived values are in sync with CD matrix
        self.update()

        image = self.rootname
        _fitsname = fitsname

        if image.find('.fits') < 0 and _fitsname is not None:
            # A non-FITS image was provided, and openImage made a copy
            # Update attributes to point to new copy instead
            self.geisname = image
            image = self.rootname = _fitsname

        # Open image as writable FITS object
        fimg = fileutil.openImage(image, mode='update', fitsname=_fitsname)

        _root,_iextn = fileutil.parseFilename(image)
        _extn = fileutil.getExtn(fimg,_iextn)

        # Write out values to header...
        if wcs:
            _wcsobj = wcs
        else:
            _wcsobj = self

        for key in _wcsobj.wcstrans.keys():
            _dkey = _wcsobj.wcstrans[key]
            if _dkey != 'pscale':
                _extn.header[key] = _wcsobj.__dict__[_dkey]

        # Close the file
        fimg.close()
        del fimg
        if archive:
            self.write_archive(fitsname=fitsname,overwrite=overwrite,quiet=quiet)