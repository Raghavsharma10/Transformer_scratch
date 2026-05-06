def enforce_cf_variable(var, mask_and_scale=True):
    """ Given a Variable constructed from GEOS-Chem output, enforce
    CF-compliant metadata and formatting.

    Until a bug with lazily-loaded data and masking/scaling is resolved in
    xarray, you have the option to manually mask and scale the data here.

    Parameters
    ----------
    var : xarray.Variable
        A variable holding information decoded from GEOS-Chem output.
    mask_and_scale : bool
        Flag to scale and mask the data given the unit conversions provided

    Returns
    -------
    out : xarray.Variable
        The original variable processed to conform to CF standards

    .. note::

        This method borrows heavily from the ideas in ``xarray.decode_cf_variable``

    """
    var = as_variable(var)
    data = var._data  # avoid loading by accessing _data instead of data
    dims = var.dims
    attrs = var.attrs.copy()
    encoding = var.encoding.copy()
    orig_dtype = data.dtype

    # Process masking/scaling coordinates. We only expect a "scale" value
    # for the units with this output.
    if 'scale' in attrs:
        scale = attrs.pop('scale')
        attrs['scale_factor'] = scale
        encoding['scale_factor'] = scale

        # TODO: Once the xr.decode_cf bug is fixed, we won't need to manually
        #       handle masking/scaling
        if mask_and_scale:
            data = scale*data

    # Process units
    # TODO: How do we want to handle parts-per-* units? These are not part of
    #       the udunits standard, and the CF conventions suggest using units
    #       like 1e-6 for parts-per-million. But we potentially mix mass and
    #       volume/molar mixing ratios in GEOS-Chem output, so we need a way
    #       to handle that edge case.
    if 'unit' in attrs:
        unit = attrs.pop('unit')
        unit = get_cfcompliant_units(unit)
        attrs['units'] = unit

    # TODO: Once the xr.decode_cf bug is fixed, we won't need to manually
    #       handle masking/scaling
    return Variable(dims, data, attrs, encoding=encoding)