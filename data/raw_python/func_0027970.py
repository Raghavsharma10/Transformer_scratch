def GetStructFormatString(self):
    """Retrieves the Python struct format string.

    Returns:
      str: format string as used by Python struct or None if format string
          cannot be determined.
    """
    if not self._element_data_type_map:
      return None

    number_of_elements = None
    if self._data_type_definition.elements_data_size:
      element_byte_size = self._element_data_type_definition.GetByteSize()
      if element_byte_size is None:
        return None

      number_of_elements, _ = divmod(
          self._data_type_definition.elements_data_size, element_byte_size)

    elif self._data_type_definition.number_of_elements:
      number_of_elements = self._data_type_definition.number_of_elements

    format_string = self._element_data_type_map.GetStructFormatString()
    if not number_of_elements or not format_string:
      return None

    return '{0:d}{1:s}'.format(number_of_elements, format_string)