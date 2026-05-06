def encode_canonical(object, output_function=None):
  """
  <Purpose>
    Encode 'object' in canonical JSON form, as specified at
    http://wiki.laptop.org/go/Canonical_JSON .  It's a restricted
    dialect of JSON in which keys are always lexically sorted,
    there is no whitespace, floats aren't allowed, and only quote
    and backslash get escaped.  The result is encoded in UTF-8,
    and the resulting bits are passed to output_function (if provided),
    or joined into a string and returned.

    Note: This function should be called prior to computing the hash or
    signature of a JSON object in TUF.  For example, generating a signature
    of a signing role object such as 'ROOT_SCHEMA' is required to ensure
    repeatable hashes are generated across different json module versions
    and platforms.  Code elsewhere is free to dump JSON objects in any format
    they wish (e.g., utilizing indentation and single quotes around object
    keys).  These objects are only required to be in "canonical JSON" format
    when their hashes or signatures are needed.

    >>> encode_canonical("")
    '""'
    >>> encode_canonical([1, 2, 3])
    '[1,2,3]'
    >>> encode_canonical([])
    '[]'
    >>> encode_canonical({"A": [99]})
    '{"A":[99]}'
    >>> encode_canonical({"x" : 3, "y" : 2})
    '{"x":3,"y":2}'

  <Arguments>
    object:
      The object to be encoded.

    output_function:
      The result will be passed as arguments to 'output_function'
      (e.g., output_function('result')).

  <Exceptions>
    securesystemslib.exceptions.FormatError, if 'object' cannot be encoded or
    'output_function' is not callable.

  <Side Effects>
    The results are fed to 'output_function()' if 'output_function' is set.

  <Returns>
    A string representing the 'object' encoded in canonical JSON form.
  """

  result = None
  # If 'output_function' is unset, treat it as
  # appending to a list.
  if output_function is None:
    result = []
    output_function = result.append

  try:
    _encode_canonical(object, output_function)

  except (TypeError, securesystemslib.exceptions.FormatError) as e:
    message = 'Could not encode ' + repr(object) + ': ' + str(e)
    raise securesystemslib.exceptions.FormatError(message)

  # Return the encoded 'object' as a string.
  # Note: Implies 'output_function' is None,
  # otherwise results are sent to 'output_function'.
  if result is not None:
    return ''.join(result)