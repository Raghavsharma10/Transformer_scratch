def generate_rsa_key(bits=_DEFAULT_RSA_KEY_BITS, scheme='rsassa-pss-sha256'):
  """
  <Purpose>
    Generate public and private RSA keys, with modulus length 'bits'.  In
    addition, a keyid identifier for the RSA key is generated.  The object
    returned conforms to 'securesystemslib.formats.RSAKEY_SCHEMA' and has the
    form:

    {'keytype': 'rsa',
     'scheme': 'rsassa-pss-sha256',
     'keyid': keyid,
     'keyval': {'public': '-----BEGIN RSA PUBLIC KEY----- ...',
                'private': '-----BEGIN RSA PRIVATE KEY----- ...'}}

    The public and private keys are strings in PEM format.

    Although the PyCA cryptography library and/or its crypto backend might set
    a minimum key size, generate() enforces a minimum key size of 2048 bits.
    If 'bits' is unspecified, a 3072-bit RSA key is generated, which is the key
    size recommended by securesystemslib.  These key size restrictions are only
    enforced for keys generated within securesystemslib.  RSA keys with sizes
    lower than what we recommended may still be imported (e.g., with
    import_rsakey_from_pem().

    >>> rsa_key = generate_rsa_key(bits=2048)
    >>> securesystemslib.formats.RSAKEY_SCHEMA.matches(rsa_key)
    True

    >>> public = rsa_key['keyval']['public']
    >>> private = rsa_key['keyval']['private']
    >>> securesystemslib.formats.PEMRSA_SCHEMA.matches(public)
    True
    >>> securesystemslib.formats.PEMRSA_SCHEMA.matches(private)
    True

  <Arguments>
    bits:
      The key size, or key length, of the RSA key.  'bits' must be 2048, or
      greater, and a multiple of 256.

    scheme:
      The signature scheme used by the key.  It must be one of
      ['rsassa-pss-sha256'].

  <Exceptions>
    securesystemslib.exceptions.FormatError, if 'bits' is improperly or invalid
    (i.e., not an integer and not at least 2048).

    ValueError, if an exception occurs after calling the RSA key generation
    routine.  The 'ValueError' exception is raised by the key generation
    function of the cryptography library called.

  <Side Effects>
    None.

  <Returns>
    A dictionary containing the RSA keys and other identifying information.
    Conforms to 'securesystemslib.formats.RSAKEY_SCHEMA'.
  """

  # Does 'bits' have the correct format?  This check will ensure 'bits'
  # conforms to 'securesystemslib.formats.RSAKEYBITS_SCHEMA'.  'bits' must be
  # an integer object, with a minimum value of 2048.  Raise
  # 'securesystemslib.exceptions.FormatError' if the check fails.
  securesystemslib.formats.RSAKEYBITS_SCHEMA.check_match(bits)
  securesystemslib.formats.RSA_SCHEME_SCHEMA.check_match(scheme)

  # Begin building the RSA key dictionary.
  rsakey_dict = {}
  keytype = 'rsa'
  public = None
  private = None

  # Generate the public and private RSA keys.  The pyca/cryptography module is
  # used to generate the actual key.  Raise 'ValueError' if 'bits' is less than
  # 1024, although a 2048-bit minimum is enforced by
  # securesystemslib.formats.RSAKEYBITS_SCHEMA.check_match().
  public, private = securesystemslib.pyca_crypto_keys.generate_rsa_public_and_private(bits)

  # When loading in PEM keys, extract_pem() is called, which strips any
  # leading or trailing new line characters. Do the same here before generating
  # the keyid.
  public =  extract_pem(public, private_pem=False)
  private = extract_pem(private, private_pem=True)

  # Generate the keyid of the RSA key.  Note: The private key material is not
  # included in the generation of the 'keyid' identifier.  Convert any '\r\n'
  # (e.g., Windows) newline characters to '\n' so that a consistent keyid is
  # generated.
  key_value = {'public': public.replace('\r\n', '\n'),
               'private': ''}
  keyid = _get_keyid(keytype, scheme, key_value)

  # Build the 'rsakey_dict' dictionary.  Update 'key_value' with the RSA
  # private key prior to adding 'key_value' to 'rsakey_dict'.
  key_value['private'] = private

  rsakey_dict['keytype'] = keytype
  rsakey_dict['scheme'] = scheme
  rsakey_dict['keyid'] = keyid
  rsakey_dict['keyid_hash_algorithms'] = securesystemslib.settings.HASH_ALGORITHMS
  rsakey_dict['keyval'] = key_value

  return rsakey_dict