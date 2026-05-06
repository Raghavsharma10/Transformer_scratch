def create_signature(key_dict, data):
  """
  <Purpose>
    Return a signature dictionary of the form:
    {'keyid': 'f30a0870d026980100c0573bd557394f8c1bbd6...',
     'sig': '...'}.

    The signing process will use the private key in
    key_dict['keyval']['private'] and 'data' to generate the signature.

    The following signature schemes are supported:

    'RSASSA-PSS'
    RFC3447 - RSASSA-PSS
    http://www.ietf.org/rfc/rfc3447.

    'ed25519'
    ed25519 - high-speed high security signatures
    http://ed25519.cr.yp.to/

    Which signature to generate is determined by the key type of 'key_dict'
    and the available cryptography library specified in 'settings'.

    >>> ed25519_key = generate_ed25519_key()
    >>> data = 'The quick brown fox jumps over the lazy dog'
    >>> signature = create_signature(ed25519_key, data)
    >>> securesystemslib.formats.SIGNATURE_SCHEMA.matches(signature)
    True
    >>> len(signature['sig'])
    128
    >>> rsa_key = generate_rsa_key(2048)
    >>> signature = create_signature(rsa_key, data)
    >>> securesystemslib.formats.SIGNATURE_SCHEMA.matches(signature)
    True
    >>> ecdsa_key = generate_ecdsa_key()
    >>> signature = create_signature(ecdsa_key, data)
    >>> securesystemslib.formats.SIGNATURE_SCHEMA.matches(signature)
    True

  <Arguments>
    key_dict:
      A dictionary containing the keys.  An example RSA key dict has the
      form:

      {'keytype': 'rsa',
       'scheme': 'rsassa-pss-sha256',
       'keyid': 'f30a0870d026980100c0573bd557394f8c1bbd6...',
       'keyval': {'public': '-----BEGIN RSA PUBLIC KEY----- ...',
                  'private': '-----BEGIN RSA PRIVATE KEY----- ...'}}

      The public and private keys are strings in PEM format.

    data:
      Data to be signed. This should be a bytes object; data should be
      encoded/serialized before it is passed here.  The same value can be be
      passed into securesystemslib.verify_signature() (along with the public
      key) to later verify the signature.

  <Exceptions>
    securesystemslib.exceptions.FormatError, if 'key_dict' is improperly
    formatted.

    securesystemslib.exceptions.UnsupportedAlgorithmError, if 'key_dict'
    specifies an unsupported key type or signing scheme.

    TypeError, if 'key_dict' contains an invalid keytype.

  <Side Effects>
    The cryptography library specified in 'settings' is called to perform the
    actual signing routine.

  <Returns>
    A signature dictionary conformant to
    'securesystemslib_format.SIGNATURE_SCHEMA'.
  """

  # Does 'key_dict' have the correct format?
  # This check will ensure 'key_dict' has the appropriate number of objects
  # and object types, and that all dict keys are properly named.
  # Raise 'securesystemslib.exceptions.FormatError' if the check fails.
  # The key type of 'key_dict' must be either 'rsa' or 'ed25519'.
  securesystemslib.formats.ANYKEY_SCHEMA.check_match(key_dict)

  # Signing the 'data' object requires a private key.  'rsassa-pss-sha256',
  # 'ed25519', and 'ecdsa-sha2-nistp256' are the only signing schemes currently
  # supported.  RSASSA-PSS keys and signatures can be generated and verified by
  # pyca_crypto_keys.py, and Ed25519 keys by PyNaCl and PyCA's optimized, pure
  # python implementation of Ed25519.
  signature = {}
  keytype = key_dict['keytype']
  scheme = key_dict['scheme']
  public = key_dict['keyval']['public']
  private = key_dict['keyval']['private']
  keyid = key_dict['keyid']
  sig = None

  if keytype == 'rsa':
    if scheme == 'rsassa-pss-sha256':
      private = private.replace('\r\n', '\n')
      sig, scheme = securesystemslib.pyca_crypto_keys.create_rsa_signature(
          private, data, scheme)

    else:
      raise securesystemslib.exceptions.UnsupportedAlgorithmError('Unsupported'
        ' RSA signature scheme specified: ' + repr(scheme))

  elif keytype == 'ed25519':
    public = binascii.unhexlify(public.encode('utf-8'))
    private = binascii.unhexlify(private.encode('utf-8'))
    sig, scheme = securesystemslib.ed25519_keys.create_signature(
        public, private, data, scheme)

  elif keytype == 'ecdsa-sha2-nistp256':
    sig, scheme = securesystemslib.ecdsa_keys.create_signature(
        public, private, data, scheme)

  # 'securesystemslib.formats.ANYKEY_SCHEMA' should have detected invalid key
  # types.  This is a defensive check against an invalid key type.
  else: # pragma: no cover
    raise TypeError('Invalid key type.')

  # Build the signature dictionary to be returned.
  # The hexadecimal representation of 'sig' is stored in the signature.
  signature['keyid'] = keyid
  signature['sig'] = binascii.hexlify(sig).decode()

  return signature