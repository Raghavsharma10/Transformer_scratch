def _parse_error_tree(error):
    """Parse an error ElementTree Node to create an ErrorInfo object

    :param error: The ElementTree error node
    :return: An ErrorInfo object containing the error ID and the message.
    """
    errinf = ErrorInfo(error.get('id'), None)
    if error.text is not None:
        errinf.message = error.text
    else:
        desc = error.find('./desc')
        if desc is not None:
            errinf.message = desc.text
    return errinf