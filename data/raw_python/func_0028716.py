def extract_pem(pem, private_pem=False):
  """
  <Purpose>
    Extract only the portion of the pem that includes the header and footer,
    with any leading and trailing characters removed.  The string returned has
    the following form:

    '-----BEGIN PUBLIC KEY----- ... -----END PUBLIC KEY-----'

    or

    '-----BEGIN RSA PRIVATE KEY----- ... -----END RSA PRIVATE KEY-----'

    Note: This function assumes "pem" is a valid pem in the following format:
    pem header + key material + key footer.  Crypto libraries (e.g., pyca's
    cryptography) that parse the pem returned by this function are expected to
    fully validate the pem.

  <Arguments>
    pem:
      A string in PEM format.

    private_pem:
      Boolean that indicates whether 'pem' is a private PEM.  Private PEMs
      are not shown in exception messages.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if 'pem' is improperly formatted.

  <Side Effects>
    Only the public and private portion of the PEM is extracted.  Leading or
    trailing whitespace is not included in the returned PEM string.

  <Returns>
    A PEM string (excluding leading and trailing newline characters).
    That is: pem header + key material + pem footer.

  """

  if private_pem:
    pem_header = '-----BEGIN RSA PRIVATE KEY-----'
    pem_footer = '-----END RSA PRIVATE KEY-----'

  else:
    pem_header = '-----BEGIN PUBLIC KEY-----'
    pem_footer = '-----END PUBLIC KEY-----'

  header_start = 0
  footer_start = 0

  # Raise error message if the expected header or footer is not found in 'pem'.
  try:
    header_start = pem.index(pem_header)

  except ValueError:
    # Be careful not to print private key material in exception message.
    if not private_pem:
      raise securesystemslib.exceptions.FormatError('Required PEM'
        ' header ' + repr(pem_header) + '\n not found in PEM'
        ' string: ' + repr(pem))

    else:
      raise securesystemslib.exceptions.FormatError('Required PEM'
        ' header ' + repr(pem_header) + '\n not found in private PEM string.')

  try:
    # Search for 'pem_footer' after the PEM header.
    footer_start = pem.index(pem_footer, header_start + len(pem_header))

  except ValueError:
    # Be careful not to print private key material in exception message.
    if not private_pem:
      raise securesystemslib.exceptions.FormatError('Required PEM'
        ' footer ' + repr(pem_footer) + '\n not found in PEM'
        ' string ' + repr(pem))

    else:
      raise securesystemslib.exceptions.FormatError('Required PEM'
        ' footer ' + repr(pem_footer) + '\n not found in private PEM string.')

  # Extract only the public portion of 'pem'.  Leading or trailing whitespace
  # is excluded.
  pem = pem[header_start:footer_start + len(pem_footer)]

  return pem