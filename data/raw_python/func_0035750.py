def name_is_valid(name):
    """Return True if the dataset name is valid.

    The name can only be 80 characters long.
    Valid characters: Alpha numeric characters [0-9a-zA-Z]
    Valid special characters: - _ .
    """
    # The name can only be 80 characters long.
    if len(name) > MAX_NAME_LENGTH:
        return False
    return bool(NAME_VALID_CHARS_REGEX.match(name))