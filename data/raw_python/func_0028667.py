def import_json():
  """
  <Purpose>
    Tries to import json module. We used to fall back to the simplejson module,
    but we have dropped support for that module. We are keeping this interface
    intact for backwards compatibility.

  <Arguments>
    None.

  <Exceptions>
    ImportError: on failure to import the json module.

  <Side Effects>
    None.

  <Return>
    json module
  """

  global _json_module

  if _json_module is not None:
    return _json_module

  else:
    try:
      module = __import__('json')

    # The 'json' module is available in Python > 2.6, and thus this exception
    # should not occur in all supported Python installations (> 2.6) of TUF.
    except ImportError: #pragma: no cover
      raise ImportError('Could not import the json module')

    else:
      _json_module = module
      return module