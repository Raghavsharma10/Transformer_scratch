def _get_exception_class_from_status_code(status_code):
    """
    Utility function that accepts a status code, and spits out a reference
    to the correct exception class to raise.

    :param str status_code: The status code to return an exception class for.
    :rtype: PetfinderAPIError or None
    :returns: The appropriate PetfinderAPIError subclass. If the status code
        is not an error, return ``None``.
    """
    if status_code == '100':
        return None

    exc_class = STATUS_CODE_MAPPING.get(status_code)
    if not exc_class:
        # No status code match, return the "I don't know wtf this is"
        # exception class.
        return STATUS_CODE_MAPPING['UNKNOWN']
    else:
        # Match found, yay.
        return exc_class