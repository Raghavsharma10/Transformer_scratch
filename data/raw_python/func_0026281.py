def from_model(cls, model_name, **kwargs):
        """
        Define a grid using the specifications of a given model.

        Parameters
        ----------
        model_name : string
            Name the model (see :func:`get_supported_models` for available
            model names).
            Supports multiple formats (e.g., 'GEOS5', 'GEOS-5' or 'GEOS_5').
        **kwargs : string
            Parameters that override the model  or default grid
          settings (See Other Parameters below).

        Returns
        -------
        A :class:`CTMGrid` object.

        Other Parameters
        ----------------
        resolution : (float, float)
            Horizontal grid resolution (lon, lat) or (DI, DJ) [degrees]
        Psurf : float
            Average surface pressure [hPa] (default: 1013.15)

        Notes
        -----
        Regridded vertical models may have several valid names (e.g.,
        'GEOS5_47L' and 'GEOS5_REDUCED' refer to the same model).

        """
        settings = _get_model_info(model_name)
        model = settings.pop('model_name')
        for k, v in list(kwargs.items()):
            if k in ('resolution', 'Psurf'):
                settings[k] = v

        return cls(model, **settings)