def CreateDataTypeMapByType(cls, data_type_definition):
    """Creates a specific data type map by type indicator.

    Args:
      data_type_definition (DataTypeDefinition): data type definition.

    Returns:
      DataTypeMap: data type map or None if the date type definition
          is not available.
    """
    data_type_map_class = cls._MAP_PER_DEFINITION.get(
        data_type_definition.TYPE_INDICATOR, None)
    if not data_type_map_class:
      return None

    return data_type_map_class(data_type_definition)