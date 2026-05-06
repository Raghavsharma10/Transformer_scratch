def from_vega(cls, **kwargs):
        """Load :ref:`Vega spectrum <synphot-vega-spec>`.

        Parameters
        ----------
        kwargs : dict
            Keywords acceptable by :func:`~synphot.specio.read_remote_spec`.

        Returns
        -------
        vegaspec : `SourceSpectrum`
            Empirical Vega spectrum.

        """
        filename = conf.vega_file
        header, wavelengths, fluxes = specio.read_remote_spec(
            filename, **kwargs)
        header['filename'] = filename
        meta = {'header': header,
                'expr': 'Vega from {0}'.format(os.path.basename(filename))}
        return cls(Empirical1D, points=wavelengths, lookup_table=fluxes,
                   meta=meta)