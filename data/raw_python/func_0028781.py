def format_base64(data):
  """
  <Purpose>
    Return the base64 encoding of 'data' with whitespace and '=' signs omitted.

  <Arguments>
    data:
      Binary or buffer of data to convert.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if the base64 encoding fails or the
    argument is invalid.

  <Side Effects>
    None.

  <Returns>
    A base64-encoded string.
  """

  try:
    return binascii.b2a_base64(data).decode('utf-8').rstrip('=\n ')

  except (TypeError, binascii.Error) as e:
    raise securesystemslib.exceptions.FormatError('Invalid base64'
      ' encoding: ' + str(e))