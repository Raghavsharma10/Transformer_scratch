def restoreWCS(self,prepend=None):
        """ Resets the WCS values to the original values stored in
            the backup keywords recorded in self.backup.
        """
        # Open header for image
        image = self.rootname

        if prepend: _prepend = prepend
        elif self.prepend: _prepend = self.prepend
        else: _prepend = None

        # Open image as writable FITS object
        fimg = fileutil.openImage(image, mode='update')
        # extract the extension ID being updated
        _root,_iextn = fileutil.parseFilename(self.rootname)
        _extn = fileutil.getExtn(fimg,_iextn)

        if len(self.backup) > 0:
            # If it knows about the backup keywords already,
            # use this to restore the original values to the original keywords
            for newkey in self.revert.keys():
                if newkey != 'opscale':
                    _orig_key = self.revert[newkey]
                    _extn.header[_orig_key] = _extn.header[newkey]
        elif _prepend:
            for key in self.wcstrans.keys():
                # Get new keyword name based on old keyname
                #    and prepend string
                if key != 'pixel scale':
                    _okey = self._buildNewKeyname(key,_prepend)

                    if _okey in _extn.header:
                        _extn.header[key] = _extn.header[_okey]
                    else:
                        print('No original WCS values found. Exiting...')
                        break
        else:
            print('No original WCS values found. Exiting...')

        fimg.close()
        del fimg