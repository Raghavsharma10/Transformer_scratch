def GetStructByteOrderString(self):
    """Retrieves the Python struct format string.

    Returns:
      str: format string as used by Python struct or None if format string
          cannot be determined.
    """
    if not self._data_type_definition:
      return None

    return self._BYTE_ORDER_STRINGS.get(
        self._data_type_definition.byte_order, None)