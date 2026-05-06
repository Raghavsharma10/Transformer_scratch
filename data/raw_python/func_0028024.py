def ReadFile(self, definitions_registry, path):
    """Reads data type definitions from a file into the registry.

    Args:
      definitions_registry (DataTypeDefinitionsRegistry): data type definitions
          registry.
      path (str): path of the file to read from.
    """
    with open(path, 'r') as file_object:
      self.ReadFileObject(definitions_registry, file_object)