def ReadFileObject(self, definitions_registry, file_object):
    """Reads data type definitions from a file-like object into the registry.

    Args:
      definitions_registry (DataTypeDefinitionsRegistry): data type definitions
          registry.
      file_object (file): file-like object to read from.

    Raises:
      FormatError: if the definitions values are missing or if the format is
          incorrect.
    """
    last_definition_object = None
    error_location = None
    error_message = None

    try:
      yaml_generator = yaml.safe_load_all(file_object)

      for yaml_definition in yaml_generator:
        definition_object = self._ReadDefinition(
            definitions_registry, yaml_definition)
        if not definition_object:
          error_location = self._GetFormatErrorLocation(
              yaml_definition, last_definition_object)
          error_message = '{0:s} Missing definition object.'.format(
              error_location)
          raise errors.FormatError(error_message)

        definitions_registry.RegisterDefinition(definition_object)
        last_definition_object = definition_object

    except errors.DefinitionReaderError as exception:
      error_message = 'in: {0:s} {1:s}'.format(
          exception.name or '<NAMELESS>', exception.message)
      raise errors.FormatError(error_message)

    except (yaml.reader.ReaderError, yaml.scanner.ScannerError) as exception:
      error_location = self._GetFormatErrorLocation({}, last_definition_object)
      error_message = '{0:s} {1!s}'.format(error_location, exception)
      raise errors.FormatError(error_message)