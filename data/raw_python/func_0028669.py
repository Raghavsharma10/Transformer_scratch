def load_json_file(filepath):
  """
  <Purpose>
    Deserialize a JSON object from a file containing the object.

  <Arguments>
    filepath:
      Absolute path of JSON file.

  <Exceptions>
    securesystemslib.exceptions.FormatError: If 'filepath' is improperly
    formatted.

    securesystemslib.exceptions.Error: If 'filepath' cannot be deserialized to
    a Python object.

    IOError in case of runtime IO exceptions.

  <Side Effects>
    None.

  <Return>
    Deserialized object.  For example, a dictionary.
  """

  # Making sure that the format of 'filepath' is a path string.
  # securesystemslib.exceptions.FormatError is raised on incorrect format.
  securesystemslib.formats.PATH_SCHEMA.check_match(filepath)

  deserialized_object = None

  # The file is mostly likely gzipped.
  if filepath.endswith('.gz'):
    logger.debug('gzip.open(' + str(filepath) + ')')
    fileobject = six.StringIO(gzip.open(filepath).read().decode('utf-8'))

  else:
    logger.debug('open(' + str(filepath) + ')')
    fileobject = open(filepath)

  try:
    deserialized_object = json.load(fileobject)

  except (ValueError, TypeError) as e:
    raise securesystemslib.exceptions.Error('Cannot deserialize to a'
      ' Python object: ' + repr(filepath))

  else:
    fileobject.close()
    return deserialized_object

  finally:
    fileobject.close()