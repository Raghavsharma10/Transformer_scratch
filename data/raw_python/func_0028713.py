def verify_signature(key_dict, signature, data):
  """
  <Purpose>
    Determine whether the private key belonging to 'key_dict' produced
    'signature'.  verify_signature() will use the public key found in
    'key_dict', the 'sig' objects contained in 'signature', and 'data' to
    complete the verification.

    >>> ed25519_key = generate_ed25519_key()
    >>> data = 'The quick brown fox jumps over the lazy dog'
    >>> signature = create_signature(ed25519_key, data)
    >>> verify_signature(ed25519_key, signature, data)
    True
    >>> verify_signature(ed25519_key, signature, 'bad_data')
    False
    >>> rsa_key = generate_rsa_key()
    >>> signature = create_signature(rsa_key, data)
    >>> verify_signature(rsa_key, signature, data)
    True
    >>> verify_signature(rsa_key, signature, 'bad_data')
    False
    >>> ecdsa_key = generate_ecdsa_key()
    >>> signature = create_signature(ecdsa_key, data)
    >>> verify_signature(ecdsa_key, signature, data)
    True
    >>> verify_signature(ecdsa_key, signature, 'bad_data')
    False

  <Arguments>
    key_dict:
      A dictionary containing the keys and other identifying information.
      If 'key_dict' is an RSA key, it has the form:

      {'keytype': 'rsa',
       'scheme': 'rsassa-pss-sha256',
       'keyid': 'f30a0870d026980100c0573bd557394f8c1bbd6...',
       'keyval': {'public': '-----BEGIN RSA PUBLIC KEY----- ...',
                  'private': '-----BEGIN RSA PRIVATE KEY----- ...'}}

      The public and private keys are strings in PEM format.

    signature:
      The signature dictionary produced by one of the key generation functions.
      'signature' has the form:

      {'keyid': 'f30a0870d026980100c0573bd557394f8c1bbd6...',
       'sig': sig}.

      Conformant to 'securesystemslib.formats.SIGNATURE_SCHEMA'.

    data:
      Data that the signature is expected to be over.  This should be a bytes
      object; data should be encoded/serialized before it is passed here.)
      This is the same value that can be passed into
      securesystemslib.create_signature() in order to create the signature.

  <Exceptions>
    securesystemslib.exceptions.FormatError, raised if either 'key_dict' or
    'signature' are improperly formatted.

    securesystemslib.exceptions.UnsupportedAlgorithmError, if 'key_dict' or
    'signature' specifies an unsupported algorithm.

    securesystemslib.exceptions.CryptoError, if the KEYID in the given
    'key_dict' does not match the KEYID in 'signature'.

  <Side Effects>
    The cryptography library specified in 'settings' called to do the actual
    verification.

  <Returns>
    Boolean.  True if the signature is valid, False otherwise.
  """

  # Does 'key_dict' have the correct format?
  # This check will ensure 'key_dict' has the appropriate number
  # of objects and object types, and that all dict keys are properly named.
  # Raise 'securesystemslib.exceptions.FormatError' if the check fails.
  securesystemslib.formats.ANYKEY_SCHEMA.check_match(key_dict)

  # Does 'signature' have the correct format?
  securesystemslib.formats.SIGNATURE_SCHEMA.check_match(signature)

  # Verify that the KEYID in 'key_dict' matches the KEYID listed in the
  # 'signature'.
  if key_dict['keyid'] != signature['keyid']:
    raise securesystemslib.exceptions.CryptoError('The KEYID ('
        ' ' + repr(key_dict['keyid']) + ' ) in the given key does not match'
        ' the KEYID ( ' + repr(signature['keyid']) + ' ) in the signature.')

  else:
    logger.debug('The KEYIDs of key_dict and the signature match.')

  # Using the public key belonging to 'key_dict'
  # (i.e., rsakey_dict['keyval']['public']), verify whether 'signature'
  # was produced by key_dict's corresponding private key
  # key_dict['keyval']['private'].
  sig = signature['sig']
  sig = binascii.unhexlify(sig.encode('utf-8'))
  public = key_dict['keyval']['public']
  keytype = key_dict['keytype']
  scheme = key_dict['scheme']
  valid_signature = False


  if keytype == 'rsa':
    if scheme == 'rsassa-pss-sha256':
      valid_signature = securesystemslib.pyca_crypto_keys.verify_rsa_signature(sig,
        scheme, public, data)

    else:
      raise securesystemslib.exceptions.UnsupportedAlgorithmError('Unsupported'
          ' signature scheme is specified: ' + repr(scheme))

  elif keytype == 'ed25519':
    if scheme == 'ed25519':
      public = binascii.unhexlify(public.encode('utf-8'))
      valid_signature = securesystemslib.ed25519_keys.verify_signature(public,
          scheme, sig, data, use_pynacl=USE_PYNACL)

    else:
      raise securesystemslib.exceptions.UnsupportedAlgorithmError('Unsupported'
          ' signature scheme is specified: ' + repr(scheme))

  elif keytype == 'ecdsa-sha2-nistp256':
    if scheme == 'ecdsa-sha2-nistp256':
      valid_signature = securesystemslib.ecdsa_keys.verify_signature(public,
        scheme, sig, data)

    else:
      raise securesystemslib.exceptions.UnsupportedAlgorithmError('Unsupported'
          ' signature scheme is specified: ' + repr(scheme))

  # 'securesystemslib.formats.ANYKEY_SCHEMA' should have detected invalid key
  # types.  This is a defensive check against an invalid key type.
  else: # pragma: no cover
    raise TypeError('Unsupported key type.')

  return valid_signature