def write(self, path, format='fits', overwrite=False, optional_kw_dict=None):
        """
        Writes the MOC to a file.

        Format can be 'fits' or 'json', though only the fits format is officially supported by the IVOA.

        Parameters
        ----------
        path : str, optional
            The path to the file to save the MOC in.
        format : str, optional
            The format in which the MOC will be serialized before being saved. Possible formats are "fits" or "json".
            By default, ``format`` is set to "fits".
        overwrite : bool, optional
            If the file already exists and you want to overwrite it, then set the  ``overwrite`` keyword. Default to False.
        optional_kw_dict : optional
            Optional keywords arguments added to the FITS header. Only used if ``format`` equals to 'fits'.
        """
        serialization = self.serialize(format=format, optional_kw_dict=optional_kw_dict)
        if format == 'fits':
            serialization.writeto(path, overwrite=overwrite)
        else:
            import json
            with open(path, 'w') as h:
                h.write(json.dumps(serialization, sort_keys=True, indent=2))