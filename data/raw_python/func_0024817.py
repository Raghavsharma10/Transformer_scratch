def get_metadata(model):
    """Get metadata for a given model.

    Parameters
    ----------
    model : `~astropy.modeling.Model`
        Model.

    Returns
    -------
    metadata : dict
        Metadata for the model.

    Raises
    ------
    synphot.exceptions.SynphotError
        Invalid model.

    """
    if not isinstance(model, Model):
        raise SynphotError('{0} is not a model.'.format(model))

    if isinstance(model, _CompoundModel):
        metadata = model._tree.evaluate(METADATA_OPERATORS, getter=None)
    else:
        metadata = deepcopy(model.meta)

    return metadata