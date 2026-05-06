def from_filter(cls, filtername, **kwargs):
        """Load :ref:`pre-defined filter bandpass <synphot-predefined-filter>`.

        Parameters
        ----------
        filtername : str
            Filter name. Choose from 'bessel_j', 'bessel_h', 'bessel_k',
            'cousins_r', 'cousins_i', 'johnson_u', 'johnson_b', 'johnson_v',
            'johnson_r', 'johnson_i', 'johnson_j', or 'johnson_k'.

        kwargs : dict
            Keywords acceptable by :func:`~synphot.specio.read_remote_spec`.

        Returns
        -------
        bp : `SpectralElement`
            Empirical bandpass.

        Raises
        ------
        synphot.exceptions.SynphotError
            Invalid filter name.

        """
        filtername = filtername.lower()

        # Select filename based on filter name
        if filtername == 'bessel_j':
            cfgitem = Conf.bessel_j_file
        elif filtername == 'bessel_h':
            cfgitem = Conf.bessel_h_file
        elif filtername == 'bessel_k':
            cfgitem = Conf.bessel_k_file
        elif filtername == 'cousins_r':
            cfgitem = Conf.cousins_r_file
        elif filtername == 'cousins_i':
            cfgitem = Conf.cousins_i_file
        elif filtername == 'johnson_u':
            cfgitem = Conf.johnson_u_file
        elif filtername == 'johnson_b':
            cfgitem = Conf.johnson_b_file
        elif filtername == 'johnson_v':
            cfgitem = Conf.johnson_v_file
        elif filtername == 'johnson_r':
            cfgitem = Conf.johnson_r_file
        elif filtername == 'johnson_i':
            cfgitem = Conf.johnson_i_file
        elif filtername == 'johnson_j':
            cfgitem = Conf.johnson_j_file
        elif filtername == 'johnson_k':
            cfgitem = Conf.johnson_k_file
        else:
            raise exceptions.SynphotError(
                'Filter name {0} is invalid.'.format(filtername))

        filename = cfgitem()

        if 'flux_unit' not in kwargs:
            kwargs['flux_unit'] = cls._internal_flux_unit

        if ((filename.endswith('fits') or filename.endswith('fit')) and
                'flux_col' not in kwargs):
            kwargs['flux_col'] = 'THROUGHPUT'

        header, wavelengths, throughput = specio.read_remote_spec(
            filename, **kwargs)
        header['filename'] = filename
        header['descrip'] = cfgitem.description
        meta = {'header': header, 'expr': filtername}
        return cls(Empirical1D, points=wavelengths, lookup_table=throughput,
                   meta=meta)