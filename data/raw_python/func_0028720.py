def is_pem_public(pem):
  """
  <Purpose>
    Checks if a passed PEM formatted string is a PUBLIC key, by looking for the
    following pattern:

    '-----BEGIN PUBLIC KEY----- ... -----END PUBLIC KEY-----'

    >>> rsa_key = generate_rsa_key()
    >>> public = rsa_key['keyval']['public']
    >>> private = rsa_key['keyval']['private']
    >>> is_pem_public(public)
    True
    >>> is_pem_public(private)
    False

  <Arguments>
    pem:
      A string in PEM format.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if 'pem' is improperly formatted.

  <Side Effects>
    None

  <Returns>
    True if 'pem' is public and false otherwise.
  """

  # Do the arguments have the correct format?
  # This check will ensure arguments have the appropriate number
  # of objects and object types, and that all dict keys are properly named.
  # Raise 'securesystemslib.exceptions.FormatError' if the check fails.
  securesystemslib.formats.PEMRSA_SCHEMA.check_match(pem)

  pem_header = '-----BEGIN PUBLIC KEY-----'
  pem_footer = '-----END PUBLIC KEY-----'

  try:
    header_start = pem.index(pem_header)
    pem.index(pem_footer, header_start + len(pem_header))

  except ValueError:
    return False

  return True