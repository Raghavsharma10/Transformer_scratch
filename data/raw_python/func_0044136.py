def write(self, filename, filetype=None, **kwargs):
        """Write the coverage data in the MOC object to a file.

        The filetype can be given or left to be inferred as for the
        read method.

        Any additional keyword arguments (kwargs) are passed on to
        the corresponding pymoc.io write functions (write_moc_fits,
        write_moc_json or write_moc_ascii).  This can be used, for
        example, to set overwrite=True (or clobber=True prior to
        Astropy version 2.0) when writing FITS files.
        """

        if filetype is not None:
            filetype = filetype.lower()
        else:
            filetype = self._guess_file_type(filename)

        if filetype == 'fits':
            from .io.fits import write_moc_fits
            write_moc_fits(self, filename, **kwargs)

        elif filetype == 'json':
            from .io.json import write_moc_json
            write_moc_json(self, filename, **kwargs)

        elif filetype == 'ascii' or filetype == 'text':
            from .io.ascii import write_moc_ascii
            write_moc_ascii(self, filename, **kwargs)

        else:
            raise ValueError('Unknown MOC file type {0}'.format(filetype))