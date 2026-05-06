def _check_encoding(name, encoding_to_check, alternative_encoding, source=None):
    """
    Check that ``encoding`` is a valid Python encoding
    :param name: name under which the encoding is known to the user, e.g. 'default encoding'
    :param encoding_to_check: name of the encoding to check, e.g. 'utf-8'
    :param source: source where the encoding has been set, e.g. option name
    :raise pygount.common.OptionError if ``encoding`` is not a valid Python encoding
    """
    assert name is not None

    if encoding_to_check not in (alternative_encoding, 'chardet', None):
        try:
            ''.encode(encoding_to_check)
        except LookupError:
            raise pygount.common.OptionError(
                '{0} is "{1}" but must be "{2}" or a known Python encoding'.format(
                    name, encoding_to_check, alternative_encoding),
                source)