def GetByteSize(self):
    """Retrieves the byte size of the data type definition.

    Returns:
      int: data type size in bytes or None if size cannot be determined.
    """
    if not self.element_data_type_definition:
      return None

    if self.elements_data_size:
      return self.elements_data_size

    if not self.number_of_elements:
      return None

    element_byte_size = self.element_data_type_definition.GetByteSize()
    if not element_byte_size:
      return None

    return element_byte_size * self.number_of_elements