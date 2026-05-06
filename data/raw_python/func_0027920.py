def truncate_rationale(rationale, max_length=MAX_RATIONALE_SIZE_IN_EVENT):
    """
    Truncates the rationale for analytics event emission if necessary

    Args:
        rationale (string): the string value of the rationale
        max_length (int): the max length for truncation

    Returns:
        truncated_value (string): the possibly truncated version of the rationale
        was_truncated (bool): returns true if the rationale is truncated

    """
    if isinstance(rationale, basestring) and max_length is not None and len(rationale) > max_length:
        return rationale[0:max_length], True
    else:
        return rationale, False