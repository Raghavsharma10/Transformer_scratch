def copy_from_model(cls, model_name, reference, **kwargs):
        """
        Set-up a user-defined grid using specifications of a reference
        grid model.

        Parameters
        ----------
        model_name : string
            name of the user-defined grid model.
        reference : string or :class:`CTMGrid` instance
            Name of the reference model (see :func:`get_supported_models`),
            or a :class:`CTMGrid` object from which grid set-up is copied.
        **kwargs
            Any set-up parameter which will override the settings of the
            reference model (see :class:`CTMGrid` parameters).

        Returns
        -------
        A :class:`CTMGrid` object.

        """
        if isinstance(reference, cls):
            settings = reference.__dict__.copy()
            settings.pop('model')
        else:
            settings = _get_model_info(reference)
            settings.pop('model_name')

        settings.update(kwargs)
        settings['reference'] = reference

        return cls(model_name, **settings)