def load_json_string(data):
  """
  <Purpose>
    Deserialize 'data' (JSON string) to a Python object.

  <Arguments>
    data:
      A JSON string.

  <Exceptions>
    securesystemslib.exceptions.Error, if 'data' cannot be deserialized to a
    Python object.

  <Side Effects>
    None.

  <Returns>
    Deserialized object.  For example, a dictionary.
  """

  deserialized_object = None

  try:
    deserialized_object = json.loads(data)

  except TypeError:
    message = 'Invalid JSON string: ' + repr(data)
    raise securesystemslib.exceptions.Error(message)

  except ValueError:
    message = 'Cannot deserialize to a Python object: ' + repr(data)
    raise securesystemslib.exceptions.Error(message)

  else:
    return deserialized_object