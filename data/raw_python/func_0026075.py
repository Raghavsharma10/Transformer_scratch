def write_archive(self,fitsname=None,overwrite=no,quiet=yes):
        """ Saves a copy of the WCS keywords from the image header
            as new keywords with the user-supplied 'prepend'
            character(s) prepended to the old keyword names.

            If the file is a GEIS image and 'fitsname' is not None, create
            a FITS copy and update that version; otherwise, raise
            an Exception and do not update anything.

        """
        _fitsname = fitsname

        # Open image in update mode
        #    Copying of GEIS images handled by 'openImage'.
        fimg = fileutil.openImage(self.rootname,mode='update',fitsname=_fitsname)
        if self.rootname.find('.fits') < 0 and _fitsname is not None:
            # A non-FITS image was provided, and openImage made a copy
            # Update attributes to point to new copy instead
            self.geisname = self.rootname
            self.rootname = _fitsname

        # extract the extension ID being updated
        _root,_iextn = fileutil.parseFilename(self.rootname)
        _extn = fileutil.getExtn(fimg,_iextn)
        if not quiet:
            print('Updating archive WCS keywords for ',_fitsname)

        # Write out values to header...
        for key in self.orig_wcs.keys():
            _comment = None
            _dkey = self.revert[key]

            # Verify that archive keywords will not be overwritten,
            # unless overwrite=yes.
            _old_key = key in _extn.header
            if  _old_key == True and overwrite == no:
                if not quiet:
                    print('WCS keyword',key,' already exists! Not overwriting.')
                continue

            # No archive keywords exist yet in file, or overwrite=yes...
            # Extract the value for the original keyword
            if _dkey in _extn.header:

                # Extract any comment string for the keyword as well
                _indx_key = _extn.header.index(_dkey)
                _full_key = _extn.header.cards[_indx_key]
                if not quiet:
                    print('updating ',key,' with value of: ',self.orig_wcs[key])
                _extn.header[key] = (self.orig_wcs[key], _full_key.comment)

        key = 'WCSCDATE'
        if key not in _extn.header:
            # Print out history keywords to record when these keywords
            # were backed up.
            _extn.header[key] = (self.orig_wcs[key], "Time WCS keywords were copied.")

        # Close the now updated image
        fimg.close()
        del fimg