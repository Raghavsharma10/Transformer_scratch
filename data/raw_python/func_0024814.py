def get_waveset(model):
    """Get optimal wavelengths for sampling a given model.

    Parameters
    ----------
    model : `~astropy.modeling.Model`
        Model.

    Returns
    -------
    waveset : array-like or `None`
        Optimal wavelengths. `None` if undefined.

    Raises
    ------
    synphot.exceptions.SynphotError
        Invalid model.

    """
    if not isinstance(model, Model):
        raise SynphotError('{0} is not a model.'.format(model))

    if isinstance(model, _CompoundModel):
        waveset = model._tree.evaluate(WAVESET_OPERATORS, getter=None)
    else:
        waveset = _get_sampleset(model)

    return waveset