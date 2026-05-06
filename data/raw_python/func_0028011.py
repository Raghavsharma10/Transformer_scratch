def _ReadLayoutDataTypeDefinition(
      self, definitions_registry, definition_values, data_type_definition_class,
      definition_name, supported_definition_values):
    """Reads a layout data type definition.

    Args:
      definitions_registry (DataTypeDefinitionsRegistry): data type definitions
          registry.
      definition_values (dict[str, object]): definition values.
      data_type_definition_class (str): data type definition class.
      definition_name (str): name of the definition.
      supported_definition_values (set[str]): names of the supported definition
          values.

    Returns:
      LayoutDataTypeDefinition: layout data type definition.

    Raises:
      DefinitionReaderError: if the definitions values are missing or if
          the format is incorrect.
    """
    return self._ReadDataTypeDefinition(
        definitions_registry, definition_values, data_type_definition_class,
        definition_name, supported_definition_values)