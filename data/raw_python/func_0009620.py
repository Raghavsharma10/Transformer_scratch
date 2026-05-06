def is_array(self, data_type):
    '''Check if a type is a known array type
    
    Args:
      data_type (str): Name of type to check
    Returns:
      True if ``data_type`` is a known array type.
    '''

    # Split off any brackets
    data_type = data_type.split('[')[0].strip()

    return data_type.lower() in self.array_types