def select_params_from_section_schema(section_schema, param_class=Param,
                                      deep=False):
    """Selects the parameters of a config section schema.

    :param section_schema:  Configuration file section schema to use.
    :return: Generator of params
    """
    # pylint: disable=invalid-name
    for name, value in inspect.getmembers(section_schema):
        if name.startswith("__") or value is None:
            continue    # pragma: no cover
        elif inspect.isclass(value) and deep:
            # -- CASE: class => SELF-CALL (recursively).
            # pylint: disable= bad-continuation
            cls = value
            for name, value in select_params_from_section_schema(cls,
                                            param_class=param_class, deep=True):
                yield (name, value)
        elif isinstance(value, param_class):
            yield (name, value)