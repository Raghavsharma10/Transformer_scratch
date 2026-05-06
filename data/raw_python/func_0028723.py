def import_ecdsakey_from_public_pem(pem, scheme='ecdsa-sha2-nistp256'):
  """
  <Purpose>
    Generate an ECDSA key object from 'pem'.  In addition, a keyid identifier
    for the ECDSA key is generated.  The object returned conforms to
    'securesystemslib.formats.ECDSAKEY_SCHEMA' and has the form:

    {'keytype': 'ecdsa-sha2-nistp256',
     'scheme': 'ecdsa-sha2-nistp256',
     'keyid': keyid,
     'keyval': {'public': '-----BEGIN PUBLIC KEY----- ...',
                'private': ''}}

    The public portion of the ECDSA key is a string in PEM format.

    >>> ecdsa_key = generate_ecdsa_key()
    >>> public = ecdsa_key['keyval']['public']
    >>> ecdsa_key['keyval']['private'] = ''
    >>> scheme = ecdsa_key['scheme']
    >>> ecdsa_key2 = import_ecdsakey_from_public_pem(public, scheme)
    >>> securesystemslib.formats.ECDSAKEY_SCHEMA.matches(ecdsa_key)
    True
    >>> securesystemslib.formats.ECDSAKEY_SCHEMA.matches(ecdsa_key2)
    True

  <Arguments>
    pem:
      A string in PEM format (it should contain a public ECDSA key).

    scheme:
      The signature scheme used by the imported key.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if 'pem' is improperly formatted.

  <Side Effects>
    Only the public portion of the PEM is extracted.  Leading or trailing
    whitespace is not included in the PEM string stored in the rsakey object
    returned.

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

  # Ensure the PEM string has a public header and footer.  Although a simple
  # validation of 'pem' is performed here, a fully valid PEM string is needed
  # later to successfully verify signatures.  Performing stricter validation of
  # PEMs are left to the external libraries that use 'pem'.

  if is_pem_public(pem):
    public_pem = extract_pem(pem, private_pem=False)

  else:
    raise securesystemslib.exceptions.FormatError('Invalid public'
        ' pem: ' + repr(pem))

  # Begin building the ECDSA key dictionary.
  ecdsakey_dict = {}
  keytype = 'ecdsa-sha2-nistp256'

  # Generate the keyid of the ECDSA key.  'key_value' corresponds to the
  # 'keyval' entry of the 'ECDSAKEY_SCHEMA' dictionary.  The private key
  # information is not included in the generation of the 'keyid' identifier.
  # Convert any '\r\n' (e.g., Windows) newline characters to '\n' so that a
  # consistent keyid is generated.
  key_value = {'public': public_pem.replace('\r\n', '\n'),
               'private': ''}
  keyid = _get_keyid(keytype, scheme, key_value)

  ecdsakey_dict['keytype'] = keytype
  ecdsakey_dict['scheme'] = scheme
  ecdsakey_dict['keyid'] = keyid
  ecdsakey_dict['keyval'] = key_value

  # Add "keyid_hash_algorithms" so that equal ECDSA keys with different keyids
  # can be associated using supported keyid_hash_algorithms.
  ecdsakey_dict['keyid_hash_algorithms'] = \
      securesystemslib.settings.HASH_ALGORITHMS

  return ecdsakey_dict