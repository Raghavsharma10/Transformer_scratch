def _get_model_info(model_name):
    """
    Get the grid specifications for a given model.

    Parameters
    ----------
    model_name : string
        Name of the model. Supports multiple formats
        (e.g., 'GEOS5', 'GEOS-5' or 'GEOS_5').

    Returns
    -------
    specifications : dict
        Grid specifications as a dictionary.

    Raises
    ------
    ValueError
        If the model is not supported (see `models`) or if the given
        `model_name` corresponds to several entries in the list of
        supported models.

    """
    # trying to get as much as possible a valid model name from the given
    # `model_name`, using regular expressions.
    split_name = re.split(r'[\-_\s]', model_name.strip().upper())
    sep_chars = ('', ' ', '-', '_')
    gen_seps = itertools.combinations_with_replacement(
        sep_chars, len(split_name) - 1
    )
    test_names = ("".join((n for n in itertools.chain(*list(zip(split_name,
                                                           s + ('',))))))
                  for s in gen_seps)
    match_names = list([name for name in test_names if name
                        in _get_supported_models()])

    if not len(match_names):
        raise ValueError("Model '{0}' is not supported".format(model_name))
    elif len(match_names) > 1:
        raise ValueError("Multiple matched models for given model name '{0}'"
                         .format(model_name))

    valid_model_name = match_names[0]
    parent_models = _find_references(valid_model_name)

    model_spec = dict()
    for m in parent_models:
        model_spec.update(MODELS[m])
    model_spec.pop('reference')
    model_spec['model_family'] = parent_models[0]
    model_spec['model_name'] = valid_model_name

    return model_spec