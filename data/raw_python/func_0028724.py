def import_ecdsakey_from_pem(pem, scheme='ecdsa-sha2-nistp256'):
  """
  <Purpose>
    Import either a public or private ECDSA PEM.  In contrast to the other
    explicit import functions (import_ecdsakey_from_public_pem and
    import_ecdsakey_from_private_pem), this function is useful for when it is
    not known whether 'pem' is private or public.

  <Arguments>
    pem:
      A string in PEM format.

    scheme:
      The signature scheme used by the imported key.
  <Exceptions>
    securesystemslib.exceptions.FormatError, if 'pem' is improperly formatted.

  <Side Effects>
    None.

  <Returns>
    A dictionary containing the ECDSA keys and other identifying information.
    Conforms to 'securesystemslib.formats.ECDSAKEY_SCHEMA'.
  """

  # Does 'pem' have the correct format?
  # This check will ensure arguments has the appropriate number
  # of objects and object types, and that all dict keys are properly named.
  # Raise 'securesystemslib.exceptions.FormatError' if the check fails.
  securesystemslib.formats.PEMECDSA_SCHEMA.check_match(pem)

  # Is 'scheme' properly formatted?
  securesystemslib.formats.ECDSA_SCHEME_SCHEMA.check_match(scheme)

  public_pem = ''
  private_pem = ''

  # Ensure the PEM string has a public or private header and footer.  Although
  # a simple validation of 'pem' is performed here, a fully valid PEM string is
  # needed later to successfully verify signatures.  Performing stricter
  # validation of PEMs are left to the external libraries that use 'pem'.
  if is_pem_public(pem):
    public_pem = extract_pem(pem, private_pem=False)

  elif is_pem_private(pem, 'ec'):
    # Return an ecdsakey object (ECDSAKEY_SCHEMA) with the private key included.
    return import_ecdsakey_from_private_pem(pem, password=None)

  else:
    raise securesystemslib.exceptions.FormatError('PEM contains neither a public'
      ' nor private key: ' + repr(pem))

  # Begin building the ECDSA key dictionary.
  ecdsakey_dict = {}
  keytype = 'ecdsa-sha2-nistp256'

  # Generate the keyid of the ECDSA key.  'key_value' corresponds to the
  # 'keyval' entry of the 'ECDSAKEY_SCHEMA' dictionary.  The private key
  # information is not included in the generation of the 'keyid' identifier.
  # If a PEM is found to contain a private key, the generated rsakey object
  # should be returned above.  The following key object is for the case of a
  # PEM with only a public key.  Convert any '\r\n' (e.g., Windows) newline
  # characters to '\n' so that a consistent keyid is generated.
  key_value = {'public': public_pem.replace('\r\n', '\n'),
               'private': ''}
  keyid = _get_keyid(keytype, scheme, key_value)

  ecdsakey_dict['keytype'] = keytype
  ecdsakey_dict['scheme'] = scheme
  ecdsakey_dict['keyid'] = keyid
  ecdsakey_dict['keyval'] = key_value

  return ecdsakey_dict