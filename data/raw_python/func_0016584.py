def _get_converter_type(identifier):
    """Return the converter type for `identifier`."""
    if isinstance(identifier, str):
        return ConverterType[identifier]
    if isinstance(identifier, ConverterType):
        return identifier
    return ConverterType(identifier)