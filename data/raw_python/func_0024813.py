def _shift_wavelengths(model1, model2):
    """One of the models is either ``RedshiftScaleFactor`` or ``Scale``.

    Possible combos::

        RedshiftScaleFactor | Model
        Scale | Model
        Model | Scale

    """
    if isinstance(model1, _models.RedshiftScaleFactor):
        val = _get_sampleset(model2)
        if val is None:
            w = val
        else:
            w = model1.inverse(val)
    elif isinstance(model1, _models.Scale):
        w = _get_sampleset(model2)
    else:
        w = _get_sampleset(model1)
    return w