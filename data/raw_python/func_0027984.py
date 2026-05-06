def GetStructFormatString(self):
    """Retrieves the Python struct format string.

    Returns:
      str: format string as used by Python struct or None if format string
          cannot be determined.
    """
    if self._format_string is None and self._data_type_maps:
      format_strings = []
      for member_data_type_map in self._data_type_maps:
        if member_data_type_map is None:
          return None

        member_format_string = member_data_type_map.GetStructFormatString()
        if member_format_string is None:
          return None

        format_strings.append(member_format_string)

      self._format_string = ''.join(format_strings)

    return self._format_string