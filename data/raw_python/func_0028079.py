def _IsIdentifier(cls, string):
    """Checks if a string contains an identifier.

    Args:
      string (str): string to check.

    Returns:
      bool: True if the string contains an identifier, False otherwise.
    """
    return (
        string and not string[0].isdigit() and
        all(character.isalnum() or character == '_' for character in string))