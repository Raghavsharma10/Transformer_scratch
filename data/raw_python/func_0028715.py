def import_rsakey_from_public_pem(pem, scheme='rsassa-pss-sha256'):
  """
  <Purpose>
    Generate an RSA key object from 'pem'.  In addition, a keyid identifier for
    the RSA key is generated.  The object returned conforms to
    'securesystemslib.formats.RSAKEY_SCHEMA' and has the form:

    {'keytype': 'rsa',
     'keyid': keyid,
     'keyval': {'public': '-----BEGIN PUBLIC KEY----- ...',
                'private': ''}}

    The public portion of the RSA key is a string in PEM format.

    >>> rsa_key = generate_rsa_key()
    >>> public = rsa_key['keyval']['public']
    >>> rsa_key['keyval']['private'] = ''
    >>> rsa_key2 = import_rsakey_from_public_pem(public)
    >>> securesystemslib.formats.RSAKEY_SCHEMA.matches(rsa_key)
    True
    >>> securesystemslib.formats.RSAKEY_SCHEMA.matches(rsa_key2)
    True

  <Arguments>
    pem:
      A string in PEM format (it should contain a public RSA key).

  <Exceptions>
    securesystemslib.exceptions.FormatError, if 'pem' is improperly formatted.

  <Side Effects>
    Only the public portion of the PEM is extracted.  Leading or trailing
    whitespace is not included in the PEM string stored in the rsakey object
    returned.

  <Returns>
    A dictionary containing the RSA keys and other identifying information.
    Conforms to 'securesystemslib.formats.RSAKEY_SCHEMA'.
  """

  # Does 'pem' have the correct format?
  # This check will ensure arguments has the appropriate number
  # of objects and object types, and that all dict keys are properly named.
  # Raise 'securesystemslib.exceptions.FormatError' if the check fails.
  securesystemslib.formats.PEMRSA_SCHEMA.check_match(pem)

  # Does 'scheme' have the correct format?
  securesystemslib.formats.RSA_SCHEME_SCHEMA.check_match(scheme)

  # Ensure the PEM string has a public header and footer.  Although a simple
  # validation of 'pem' is performed here, a fully valid PEM string is needed
  # later to successfully verify signatures.  Performing stricter validation of
  # PEMs are left to the external libraries that use 'pem'.

  if is_pem_public(pem):
    public_pem = extract_pem(pem, private_pem=False)

  else:
    raise securesystemslib.exceptions.FormatError('Invalid public'
        ' pem: ' + repr(pem))

  # Begin building the RSA key dictionary.
  rsakey_dict = {}
  keytype = 'rsa'

  # Generate the keyid of the RSA key.  'key_value' corresponds to the
  # 'keyval' entry of the 'RSAKEY_SCHEMA' dictionary.  The private key
  # information is not included in the generation of the 'keyid' identifier.
  # Convert any '\r\n' (e.g., Windows) newline characters to '\n' so that a
  # consistent keyid is generated.
  key_value = {'public': public_pem.replace('\r\n', '\n'),
               'private': ''}
  keyid = _get_keyid(keytype, scheme, key_value)

  rsakey_dict['keytype'] = keytype
  rsakey_dict['scheme'] = scheme
  rsakey_dict['keyid'] = keyid
  rsakey_dict['keyval'] = key_value

  # Add "keyid_hash_algorithms" so that equal RSA keys with different keyids
  # can be associated using supported keyid_hash_algorithms.
  rsakey_dict['keyid_hash_algorithms'] = \
      securesystemslib.settings.HASH_ALGORITHMS

  return rsakey_dict