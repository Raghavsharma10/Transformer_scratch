def GetStructFormatString(self):
    """Retrieves the Python struct format string.

    Returns:
      str: format string as used by Python struct or None if format string
          cannot be determined.
    """
    if self._data_type_definition.format == definitions.FORMAT_UNSIGNED:
      return self._FORMAT_STRINGS_UNSIGNED.get(
          self._data_type_definition.size, None)

    return self._FORMAT_STRINGS_SIGNED.get(
        self._data_type_definition.size, None)