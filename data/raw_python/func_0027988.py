def CreateDataTypeMap(self, definition_name):
    """Creates a specific data type map by name.

    Args:
      definition_name (str): name of the data type definition.

    Returns:
      DataTypeMap: data type map or None if the date type definition
          is not available.
    """
    data_type_definition = self._definitions_registry.GetDefinitionByName(
        definition_name)
    if not data_type_definition:
      return None

    return DataTypeMapFactory.CreateDataTypeMapByType(data_type_definition)