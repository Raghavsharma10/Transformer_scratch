def list_to_csv(value):
    """
    Converts list to string with comma separated values. For string is no-op.
    """
    if isinstance(value, (list, tuple, set)):
        value = ",".join(value)
    return value