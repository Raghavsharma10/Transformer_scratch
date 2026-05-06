def add_element(source, path, value, separator=r'[/.]', **kwargs):
    """
    Add element into a list or dict easily using a path.
    =============   =============   =======================================
    Parameter       Type            Description
    =============   =============   =======================================
    source          list or dict    element where add the value.
    path            string          path to add the value in element.
    value           ¿all?           value to add in source.
    separator       regex string    Regexp to divide the path.
    =============   =============   =======================================
    Returns: source with added value
    """

    return _add_element_by_names(
        source,
        exclude_empty_values(re.split(separator, path)),
        value,
        **kwargs)