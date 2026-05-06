def IsComposite(self):
    """Determines if the data type is composite.

    A composite data type consists of other data types.

    Returns:
      bool: True if the data type is composite, False otherwise.
    """
    return bool(self.condition) or (
        self.member_data_type_definition and
        self.member_data_type_definition.IsComposite())