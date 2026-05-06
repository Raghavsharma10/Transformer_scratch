def is_pem_private(pem, keytype='rsa'):
  """
  <Purpose>
    Checks if a passed PEM formatted string is a PRIVATE key, by looking for
    the following patterns:

    '-----BEGIN RSA PRIVATE KEY----- ... -----END RSA PRIVATE KEY-----'
    '-----BEGIN EC PRIVATE KEY----- ... -----END EC PRIVATE KEY-----'

    >>> rsa_key = generate_rsa_key()
    >>> private = rsa_key['keyval']['private']
    >>> public = rsa_key['keyval']['public']
    >>> is_pem_private(private)
    True
    >>> is_pem_private(public)
    False

  <Arguments>
    pem:
      A string in PEM format.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if any of the arguments are
    improperly formatted.

  <Side Effects>
    None

  <Returns>
    True if 'pem' is private and false otherwise.
  """

  # Do the arguments have the correct format?
  # This check will ensure arguments have the appropriate number
  # of objects and object types, and that all dict keys are properly named.
  # Raise 'securesystemslib.exceptions.FormatError' if the check fails.
  securesystemslib.formats.PEMRSA_SCHEMA.check_match(pem)
  securesystemslib.formats.NAME_SCHEMA.check_match(keytype)

  if keytype == 'rsa':
    pem_header = '-----BEGIN RSA PRIVATE KEY-----'
    pem_footer = '-----END RSA PRIVATE KEY-----'

  elif keytype == 'ec':
    pem_header = '-----BEGIN EC PRIVATE KEY-----'
    pem_footer = '-----END EC PRIVATE KEY-----'

  else:
    raise securesystemslib.exceptions.FormatError('Unsupported key'
      ' type: ' + repr(keytype) + '.  Supported keytypes: ["rsa", "ec"]')

  try:
    header_start = pem.index(pem_header)
    pem.index(pem_footer, header_start + len(pem_header))

  except ValueError:
    return False

  return True