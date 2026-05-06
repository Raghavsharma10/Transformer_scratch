def read(self, filename, filetype=None, include_meta=False, **kwargs):
        """Read data from the given file into the MOC object.

        The cell lists read from the file are added to the current
        object.  Therefore if the object already contains some
        cells, it will be updated to represent the union of the
        current coverge and that from the file.

        The file type can be specified as "fits", "json" or "ascii",
        with "text" allowed as an alias for "ascii".  If the type
        is not specified, then an attempt will be made to guess
        from the file name, or the contents of the file.

        Note that writing to FITS and JSON will cause the MOC
        to be normalized automatically.

        Any additional keyword arguments (kwargs) are passed on to
        the corresponding pymoc.io read functions (read_moc_fits,
        read_moc_json or read_moc_ascii).
        """

        if filetype is not None:
            filetype = filetype.lower()
        else:
            filetype = self._guess_file_type(filename)

        if filetype == 'fits':
            from .io.fits import read_moc_fits
            read_moc_fits(self, filename, include_meta, **kwargs)

        elif filetype == 'json':
            from .io.json import read_moc_json
            read_moc_json(self, filename, **kwargs)

        elif filetype == 'ascii' or filetype == 'text':
            from .io.ascii import read_moc_ascii
            read_moc_ascii(self, filename, **kwargs)

        else:
            raise ValueError('Unknown MOC file type {0}'.format(filetype))