def GetDefinitionByName(self, name):
    """Retrieves a specific data type definition by name.

    Args:
      name (str): name of the data type definition.

    Returns:
      DataTypeDefinition: data type definition or None if not available.
    """
    lookup_name = name.lower()
    if lookup_name not in self._definitions:
      lookup_name = self._aliases.get(name, None)

    return self._definitions.get(lookup_name, None)