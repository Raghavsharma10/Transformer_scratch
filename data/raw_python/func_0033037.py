def _candidate_type_names(python_type_representation):
    """Generator which yields possible type names to look up in the conversion
    dictionary.

    Parameters
    ----------
    python_type_representation : object
        Any Python object which represents a type, such as `int`,
        `dtype('int8')`, `np.int8`, or `"int8"`.
    """
    # if we get a single character code we should normalize to a NumPy type
    # using np.typeDict, which maps string representations of types to NumPy
    # type objects
    if python_type_representation in np.typeDict:
        python_type_representation = np.typeDict[python_type_representation]
        yield python_type_representation.__name__

    # if we get a dtype object i.e. dtype('int16'), then pull out its name
    if hasattr(python_type_representation, 'name'):
        yield python_type_representation.name

    # convert Python types by adding their type's name
    if hasattr(python_type_representation, '__name__'):
        yield python_type_representation.__name__

    # for a dtype like dtype('S3') need to access dtype.type.__name__
    # to get 'string_'
    if hasattr(python_type_representation, 'type'):
        if hasattr(python_type_representation.type, '__name__'):
            yield python_type_representation.type.__name__

    yield str(python_type_representation)